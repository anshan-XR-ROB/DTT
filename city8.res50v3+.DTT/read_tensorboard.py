
from tensorboard.backend.event_processing import event_accumulator     
 
ea=event_accumulator.EventAccumulator("/TorchSemiSeg-main/exp.voc/voc8.res50v3+.CPS/log/tb/Jun24_24-20-26/events.out.tfevents.1624537615.A01-R04-I231-41-7276589.JD.LOCAL")     # 初始化EventAccumulator对象
ea.Reload()  
print(ea.scalars.Keys())   
