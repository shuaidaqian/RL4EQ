"""
完整的PPO训练流程实现
包含: Actor, Reference, Critic, Reward 四个核心模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import deque
import copy


# ==================== 配置类 ====================

@dataclass
class PPOConfig:
    """PPO训练配置参数"""
    # 模型路径
    actor_model_path: str = "meta-llama/Llama-2-7b-hf"
    reward_model_path: str = "OpenAssistant/reward-model-deberta-v3-large"
    
    # 训练参数
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    ppo_epochs: int = 4  # 每次数据重复训练次数
    num_episodes: int = 1000  # 总训练轮数
    
    # PPO特定参数
    clip_epsilon: float = 0.2  # PPO裁剪阈值
    value_clip: float = 0.4    # Value function裁剪
    beta: float = 0.1          # KL散度惩罚系数
    gamma: float = 0.99        # 折扣因子
    gae_lambda: float = 0.95   # GAE lambda参数
    
    # 生成参数
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 经验回放缓冲区 ====================

class ExperienceBuffer:
    """
    存储PPO训练所需的经验数据
    每个样本包含: query, response, 旧策略概率, 奖励, 价值估计等
    """
    def __init__(self):
        self.experiences = []
    
    def add(self, experience: Dict):
        """添加一条经验"""
        self.experiences.append(experience)
    
    def clear(self):
        """清空缓冲区"""
        self.experiences = []
    
    def get_batch(self) -> List[Dict]:
        """获取所有经验"""
        return self.experiences
    
    def __len__(self):
        return len(self.experiences)


# ==================== 核心模型定义 ====================

class ActorModel(nn.Module):
    """
    Actor模型: 策略网络，负责生成文本
    这是我们要训练的主模型
    """
    def __init__(self, model_path: str, device: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
        # 可训练参数
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """前向传播，返回logits"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits
    
    def generate_with_logprobs(
        self, 
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成回复，并计算每个token的log概率
        这是PPO的关键：需要知道旧策略下的动作概率
        """
        batch_size = query_ids.size(0)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=query_ids,
                attention_mask=query_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # 分离query和response
        response_ids = generated_ids.sequences[:, query_ids.size(1):]
        full_ids = generated_ids.sequences
        
        # 重新计算log概率（用于存储旧策略）
        full_mask = torch.ones_like(full_ids)
        full_mask[:, :query_ids.size(1)] = query_mask
        
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            return_dict=True
        )
        
        # 计算每个token的log概率
        logits = outputs.logits[:, :-1, :]  # 去掉最后一个
        target_ids = full_ids[:, 1:]        # 偏移一位
        
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # 只取response部分的log概率
        response_log_probs = token_log_probs[:, query_ids.size(1)-1:]
        
        return full_ids, response_ids, response_log_probs
    
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算给定response在当前策略下的log概率
        用于PPO更新时计算新策略概率
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # 只计算response部分的log概率
        query_length = input_ids.size(1) - response_ids.size(1)
        response_log_probs = token_log_probs[:, query_length-1:]
        
        return response_log_probs


class ReferenceModel(nn.Module):
    """
    Reference模型: 参考策略，通常是SFT后的模型
    作用: 1) 计算KL散度惩罚 2) 防止Actor偏离太远
    特点: 冻结参数，不参与训练
    """
    def __init__(self, model_path: str, device: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.eval()  # 设置为评估模式
    
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Reference模型下的log概率
        用于KL散度计算
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            logits = outputs.logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                dim=2,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            query_length = input_ids.size(1) - response_ids.size(1)
            response_log_probs = token_log_probs[:, query_length-1:]
            
        return response_log_probs


class CriticModel(nn.Module):
    """
    Critic模型: 价值网络，估计状态价值V(s)
    作用: 用于计算优势函数A(s,a) = Q(s,a) - V(s)
    通常与Actor共享部分结构，但输出头不同
    """
    def __init__(self, model_path: str, device: str):
        super().__init__()
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 使用语言模型的隐藏层
        self.transformer = base_model.model
        
        # 添加价值头：将隐藏状态映射到标量价值
        hidden_size = self.transformer.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # 输出单一价值
        )
        
        # 初始化价值头
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        
        self.device = device
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个token位置的状态价值V(s)
        输出形状: [batch_size, seq_length]
        """
        # 获取隐藏状态
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # 计算价值
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq]
        
        return values
    
    def get_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_length: int
    ) -> torch.Tensor:
        """
        获取response部分的价值估计
        """
        values = self.forward(input_ids, attention_mask)
        query_length = input_ids.size(1) - response_length
        response_values = values[:, query_length-1:-1]  # 对应response位置
        return response_values


class RewardModel(nn.Module):
    """
    Reward模型: 奖励网络，评估生成质量
    通常是基于人类偏好数据训练的BERT类模型
    输出标量奖励分数
    """
    def __init__(self, model_path: str, device: str):
        super().__init__()
        # Reward模型通常是编码器模型（如DeBERTa）
        from transformers import AutoModelForSequenceClassification
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,  # 回归任务，输出单一分数
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 冻结参数（通常Reward模型是固定的）
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.eval()
        self.device = device
    
    def get_rewards(
        self,
        query_texts: List[str],
        response_texts: List[str]
    ) -> torch.Tensor:
        """
        计算query-response对的奖励分数
        """
        # 构建输入
        inputs = self.tokenizer(
            query_texts,
            response_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            rewards = outputs.logits.squeeze(-1)  # [batch_size]
        
        return rewards


# ==================== PPO训练器 ====================

class PPOTrainer:
    """
    PPO训练器：协调四个模型，执行完整的PPO训练流程
    """
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device
        
        # 初始化四个核心模型
        print("🚀 初始化PPO模型...")
        
        # 1. Actor模型（需要训练）
        print("  加载 Actor Model...")
        self.actor = ActorModel(config.actor_model_path, self.device)
        
        # 2. Reference模型（冻结，用于KL惩罚）
        print("  加载 Reference Model...")
        self.reference = ReferenceModel(config.actor_model_path, self.device)
        
        # 3. Critic模型（需要训练）
        print("  加载 Critic Model...")
        self.critic = CriticModel(config.actor_model_path, self.device)
        
        # 4. Reward模型（冻结，提供外部奖励信号）
        print("  加载 Reward Model...")
        self.reward_model = RewardModel(config.reward_model_path, self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.model.parameters(),
            lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        
        # 经验缓冲区
        self.buffer = ExperienceBuffer()
        
        # 训练统计
        self.global_step = 0
    
    # ==================== Step 1: 生成阶段 ====================
    def generate_experiences(self, queries: List[str]) -> List[Dict]:
        """
        【第一步】生成阶段：Actor生成回复，收集经验
        
        执行流程：
        1. Actor模型根据query生成response
        2. 同时记录旧策略下的log概率（old_log_probs）
        3. 存储query、response、old_log_probs等
        
        目的：
        - 收集训练数据（经验）
        - 记录旧策略分布，用于后续的PPO比率计算
        """
        experiences = []
        
        for query in queries:
            # Tokenize query
            query_tokens = self.actor.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            query_ids = query_tokens["input_ids"]
            query_mask = query_tokens["attention_mask"]
            
            # Actor生成回复，并获取旧策略的log概率
            with torch.no_grad():
                full_ids, response_ids, old_log_probs = self.actor.generate_with_logprobs(
                    query_ids=query_ids,
                    query_mask=query_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
                
                # 解码文本（用于Reward模型）
                response_text = self.actor.tokenizer.decode(
                    response_ids[0], 
                    skip_special_tokens=True
                )
            
            experience = {
                "query": query,
                "query_ids": query_ids,
                "query_mask": query_mask,
                "response_ids": response_ids,
                "full_ids": full_ids,
                "old_log_probs": old_log_probs,  # 关键：旧策略概率
                "response_text": response_text,
                "response_length": response_ids.size(1)
            }
            experiences.append(experience)
        
        return experiences
    
    # ==================== Step 2: 奖励计算阶段 ====================
    def compute_rewards(self, experiences: List[Dict]) -> List[Dict]:
        """
        【第二步】奖励计算阶段：计算综合奖励
        
        执行流程：
        1. Reward模型评估生成质量 → r_reward
        2. Reference模型计算参考策略概率 → ref_log_probs
        3. 计算KL散度惩罚：KL = old_log_probs - ref_log_probs
        4. 最终奖励 = r_reward - beta * KL
        
        目的：
        - 结合外部奖励（Reward模型）和内部约束（KL惩罚）
        - 防止Actor策略偏离Reference太远（模式崩溃）
        """
        queries = [exp["query"] for exp in experiences]
        responses = [exp["response_text"] for exp in experiences]
        
        # 1. 获取Reward模型的奖励
        rewards = self.reward_model.get_rewards(queries, responses)
        
        # 2. 计算KL散度惩罚
        for i, exp in enumerate(experiences):
            with torch.no_grad():
                # Reference模型概率
                ref_log_probs = self.reference.get_logprobs(
                    exp["full_ids"],
                    torch.ones_like(exp["full_ids"]),
                    exp["response_ids"]
                )
                
                # KL散度（逐token）
                kl_penalty = exp["old_log_probs"] - ref_log_probs
                kl_penalty = kl_penalty.squeeze(0)  # [response_length]
                
                # 总KL惩罚（取平均）
                kl_mean = kl_penalty.mean()
                
                # 3. 最终奖励（只在最后一个token给奖励，中间为0）
                final_reward = rewards[i] - self.config.beta * kl_mean
                
                # 构建奖励序列：前面为0，最后一个为final_reward
                reward_seq = torch.zeros_like(exp["old_log_probs"].squeeze(0))
                reward_seq[-1] = final_reward
                
                exp["rewards"] = reward_seq
                exp["kl_penalty"] = kl_penalty
                exp["raw_reward"] = rewards[i]
        
        return experiences
    
    # ==================== Step 3: 优势估计阶段 ====================
    def compute_advantages(self, experiences: List[Dict]) -> List[Dict]:
        """
        【第三步】优势估计阶段：使用GAE计算优势函数
        
        执行流程：
        1. Critic模型计算每个token的价值V(s)
        2. 使用GAE（广义优势估计）计算优势A(s,a)
        3. 同时计算returns（用于更新Critic）
        
        公式：
        - TD残差：δ_t = r_t + γV(s_{t+1}) - V(s_t)
        - GAE优势：Â_t = Σ(γλ)^l * δ_{t+l}
        - Returns：R_t = Â_t + V(s_t)
        
        目的：
        - 减少价值估计的方差
        - 提供稳定的策略梯度信号
        """
        for exp in experiences:
            with torch.no_grad():
                # 1. Critic评估价值
                values = self.critic.get_values(
                    exp["full_ids"],
                    torch.ones_like(exp["full_ids"]),
                    exp["response_length"]
                ).squeeze(0)  # [response_length]
                
                # 添加最后一个价值的估计（用于bootstrap）
                # 简化处理：假设最后一个next_value = 0
                values_np = values.cpu().numpy()
                rewards_np = exp["rewards"].cpu().numpy()
                
                # 2. GAE计算
                advantages = []
                gae = 0
                
                # 从后向前计算
                for t in reversed(range(len(rewards_np))):
                    if t == len(rewards_np) - 1:
                        next_value = 0  # 终止状态
                    else:
                        next_value = values_np[t + 1]
                    
                    # TD残差
                    delta = rewards_np[t] + self.config.gamma * next_value - values_np[t]
                    
                    # GAE累加
                    gae = delta + self.config.gamma * self.config.gae_lambda * gae
                    advantages.insert(0, gae)
                
                advantages = torch.tensor(advantages, device=self.device)
                
                # 3. 计算returns（优势+价值）
                returns = advantages + values
                
                # 标准化优势（有助于训练稳定）
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                exp["values"] = values
                exp["advantages"] = advantages
                exp["returns"] = returns
        
        return experiences
    
    # ==================== Step 4: 策略更新阶段 ====================
    def update_policy(self, experiences: List[Dict]) -> Dict:
        """
        【第四步】策略更新阶段：更新Actor和Critic
        
        执行流程：
        1. 对每个mini-batch：
           - 重新计算当前策略的log概率（new_log_probs）
           - 计算概率比率：ratio = exp(new_log_probs - old_log_probs)
           - 计算PPO裁剪损失：L^CLIP
           - 计算价值损失：MSE(V_pred, V_target)
           - 计算总损失并反向传播
        
        2. 重复ppo_epochs次（数据复用）
        
        目的：
        - 在信任区域内更新策略（防止破坏性更新）
        - 最大化优势函数同时保持策略稳定性
        """
        # 准备数据
        all_full_ids = torch.cat([exp["full_ids"] for exp in experiences])
        all_old_log_probs = torch.cat([exp["old_log_probs"] for exp in experiences])
        all_advantages = torch.cat([exp["advantages"] for exp in experiences])
        all_returns = torch.cat([exp["returns"] for exp in experiences])
        all_response_lengths = [exp["response_length"] for exp in experiences]
        
        # 记录训练指标
        metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "kl_div": [],
            "clip_fraction": []
        }
        
        # PPO多轮更新
        for ppo_epoch in range(self.config.ppo_epochs):
            # 生成随机索引进行mini-batch训练
            indices = torch.randperm(len(experiences))
            
            for i in range(0, len(experiences), self.config.mini_batch_size):
                batch_indices = indices[i:i + self.config.mini_batch_size]
                
                # 获取batch数据
                batch_experiences = [experiences[idx] for idx in batch_indices]
                
                # 拼接batch
                batch_full_ids = torch.cat([exp["full_ids"] for exp in batch_experiences])
                batch_attention_mask = torch.ones_like(batch_full_ids)
                batch_old_log_probs = torch.cat([exp["old_log_probs"] for exp in batch_experiences])
                batch_advantages = torch.cat([exp["advantages"] for exp in batch_experiences])
                batch_returns = torch.cat([exp["returns"] for exp in batch_experiences])
                
                # ===== Actor更新 =====
                self.actor_optimizer.zero_grad()
                
                # 新策略下的log概率
                new_log_probs_list = []
                for exp in batch_experiences:
                    log_probs = self.actor.get_logprobs(
                        exp["full_ids"],
                        torch.ones_like(exp["full_ids"]),
                        exp["response_ids"]
                    )
                    new_log_probs_list.append(log_probs)
                
                new_log_probs = torch.cat(new_log_probs_list)
                
                # 计算比率 ratio = π_new / π_old
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_epsilon, 
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                
                # PPO损失（取最小值，防止过度优化）
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 添加KL惩罚项（可选，增强稳定性）
                kl_div = (batch_old_log_probs - new_log_probs).mean()
                
                total_actor_loss = actor_loss + 0.01 * kl_div
                total_actor_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.model.parameters(), 1.0)
                self.actor_optimizer.step()
                
                # ===== Critic更新 =====
                self.critic_optimizer.zero_grad()
                
                # 新的价值估计
                new_values_list = []
                for exp in batch_experiences:
                    values = self.critic.get_values(
                        exp["full_ids"],
                        torch.ones_like(exp["full_ids"]),
                        exp["response_length"]
                    )
                    new_values_list.append(values)
                
                new_values = torch.cat(new_values_list)
                
                # 价值损失（MSE）
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # 价值裁剪（可选，模仿PPO的稳定性）
                value_clip = self.config.value_clip
                with torch.no_grad():
                    old_values = torch.cat([exp["values"] for exp in batch_experiences])
                
                value_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -value_clip,
                    value_clip
                )
                value_loss_clipped = F.mse_loss(value_clipped, batch_returns)
                value_loss_final = torch.max(value_loss, value_loss_clipped)
                
                value_loss_final.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
                
                # 记录指标
                clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
                
                metrics["actor_loss"].append(actor_loss.item())
                metrics["critic_loss"].append(value_loss.item())
                metrics["kl_div"].append(kl_div.item())
                metrics["clip_fraction"].append(clip_fraction.item())
        
        # 返回平均指标
        return {k: np.mean(v) for k, v in metrics.items()}
    
    # ==================== 完整训练循环 ====================
    def train(self, queries: List[str]):
        """
        完整的PPO训练流程
        
        循环：
        for episode in range(num_episodes):
            1. generate_experiences()   # 生成
            2. compute_rewards()        # 奖励
            3. compute_advantages()     # 优势
            4. update_policy()          # 更新
        """
        print(f"\n🎯 开始PPO训练，共 {self.config.num_episodes} 个episodes")
        
        for episode in range(self.config.num_episodes):
            print(f"\n📦 Episode {episode + 1}/{self.config.num_episodes}")
            
            # Step 1: 生成经验
            print("  Step 1/4: 生成经验...")
            experiences = self.generate_experiences(queries)
            print(f"    生成了 {len(experiences)} 条经验")
            
            # Step 2: 计算奖励
            print("  Step 2/4: 计算奖励...")
            experiences = self.compute_rewards(experiences)
            avg_reward = torch.stack([exp["raw_reward"] for exp in experiences]).mean().item()
            print(f"    平均奖励: {avg_reward:.4f}")
            
            # Step 3: 计算优势
            print("  Step 3/4: 计算优势...")
            experiences = self.compute_advantages(experiences)
            
            # Step 4: 更新策略
            print("  Step 4/4: 更新策略...")
            metrics = self.update_policy(experiences)
            
            print(f"    Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"    Critic Loss: {metrics['critic_loss']:.4f}")
            print(f"    KL散度: {metrics['kl_div']:.4f}")
            print(f"    裁剪比例: {metrics['clip_fraction']:.2%}")
            
            self.global_step += 1
            
            # 定期保存
            if (episode + 1) % 100 == 0:
                self.save_checkpoint(f"checkpoint_episode_{episode+1}")
        
        print("\n✅ 训练完成！")
    
    def save_checkpoint(self, path: str):
        """保存模型检查点"""
        torch.save({
            "actor": self.actor.model.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "global_step": self.global_step
        }, path)
        print(f"💾 检查点已保存: {path}")


# ==================== 使用示例 ====================

def main():
    # 配置
    config = PPOConfig(
        actor_model_path="gpt2",  # 使用小模型演示
        reward_model_path="gpt2",  # 实际应使用专门的reward模型
        batch_size=2,
        num_episodes=10
    )
    
    # 示例查询（实际应为训练数据集）
    sample_queries = [
        "请介绍一下人工智能的发展历程",
        "解释一下量子计算的基本原理",
        "如何学习Python编程？",
        "什么是深度学习？"
    ]
    
    # 初始化训练器
    trainer = PPOTrainer(config)
    
    # 开始训练
    trainer.train(sample_queries)


if __name__ == "__main__":
    main()