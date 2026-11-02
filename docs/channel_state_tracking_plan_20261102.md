# medium 状态动作的 128 帧独立持续跟踪验证

新的训练分布 78000–78002 中，likelihood medium 经原 guard 后在 0 dB 有探索性
正区间，MAP 没有建立统一收益。固定一次持续会话验证，不根据部分留出调参。

- 原 A/B channel_state 256 checkpoint 不变，SHA 分别
  `b7d6e865abc34a4e3aa8a26dbd3455fa0c7a4fac7bd6a802035a322f6269150f`、
  `0fdd233cd406dced2ec3ec75f18933671ec42651f88a0c37dd70ae081156c490`。
- heldout_edge 84001–84003、四 SNR、各 128 帧，共 1536 上下文、3072 条 A/B 记录。
- medium 固定 lr=.02、3 步、累计 head 半径 .2，目标为 Adapt signal likelihood。
  训练时内层为 .005 一步、半径 .05，因此此轮是动作幅度迁移检查，不能称完全
  匹配训练的四臂主结果，也不能据此单独归因元训练作用。
- Adapt 更新、Reward BER/前向信号 loss 验收；原 guard 不变。不使用 MAP。
  接受后跨帧保留、拒绝保留动作前参数，每个 seed 从离线参数重置。
- Frozen 与 Online 同 checkpoint、同波形、同帧龄，状态支路都读取当前合法 Adapt。
- 早期固定 0–41，后期 86–127；统计单位为独立 seed，分别报告平均和后期优势。
- 用户选择的持续跟踪目标保持不变，但本轮仅 13.1072 秒，不能代替数百秒验证。
- 这仍是固定动作诊断。正式 RL、校准传统比较以及四 SNR 最终门槛均未通过。

若只在部分 SNR 有效，应记录其边界，先检验合法 Pilot 反馈与状态是否足够
区分有益/有害动作，不用扩大相同 seed 矩阵或选最好的 SNR 宣称目标完成。
