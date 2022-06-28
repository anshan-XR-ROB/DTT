
from tensorboard.backend.event_processing import event_accumulator        # 导入tensorboard的事件解析器
 
ea=event_accumulator.EventAccumulator("TorchSemiSeg-main/exp.voc/voc8.res50v3+.CPS/log/tb/Jun24_24-20-26/events.out.tfevents.1624537615.A01-R04-I231-41-7276589.JD.LOCAL")     # 初始化EventAccumulator对象
ea.Reload()    # 这一步是必须的，将事件的内容都导进去
print(ea.scalars.Keys())    # 我们知道tensorboard可以保存Image scalars等对象，我们主要关注scalars
# train_loss = ea.scalars.Items("train_loss")    # 读取train_loss