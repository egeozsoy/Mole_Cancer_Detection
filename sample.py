from fastai.vision import *
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learner = cnn_learner(data, models.resnet18, metrics=accuracy)
learner.lr_find()
fig = learner.recorder.plot(return_fig=True)
plt.show()

learner.fit(1)
a = learner.get_preds()[1]
print(a)