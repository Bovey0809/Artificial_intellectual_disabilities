# Use train sample from torchsample to train.
from torchsample.modules import ModuleTrainer
from model import CharRNN


model = CharRNN()
trainer = ModuleTrainer(model)
trainer.compile(loss='CrossEntropyLoss')


trainer.fit(x_train, y_train,
            val_data=(x_test, y_test),
            num_epoch=20,
            batch_size=128,
            cuda_device=1)
