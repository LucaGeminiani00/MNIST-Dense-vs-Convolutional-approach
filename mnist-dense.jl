using MLDatasets: MNIST
using Flux:crossentropy, onecold, onehotbatch, train!
using Plots, Images
using LinearAlgebra, Random, Statistics 

train_images_raw, train_labels_raw = MNIST.traindata(Float32)
test_images_raw, test_labels_raw = MNIST.testdata(Float32)

index = 8
img = train_images_raw[:,:,index]
colorview(Gray, img')
train_labels_raw[index] #check that the image actually corresponds to the index

#Flatten the data and use onehotencoding on the labels 

train_images = Flux.flatten(train_images_raw)
#train_images ./ 255 (Non necessario)
test_images = Flux.flatten(test_images_raw)
#test_images ./ 255
train_labels = onehotbatch(train_labels_raw, 0:9)
test_labels = onehotbatch(test_labels_raw, 0:9)



model = Chain(
    Dense(28*28, 512, relu),
    Dense(512, 10), 
    softmax
    )

loss(x,y) = crossentropy(model(x),y)
ps = Flux.params(model)
opt = Flux.RMSProp()

epochs = 100
loss_history = []
for epoch in 1:epochs
    train!(loss, ps, [(train_images,train_labels)], opt)
    train_loss = loss(train_images, train_labels)
    push!(loss_history, train_loss)
    println("Epoch = $epoch Loss =$train_loss")
end 

estimated_images_raw = model(test_images)
estimated_images = onecold(estimated_images_raw) .-1 #because also the first value is included 

mean(estimated_images .== test_labels_raw) 

#Plotting the learning curve 

p_l_curve = plot(1:epochs, loss_history,xlabel = "Epochs", ylabel = "Loss", 
title = "Learning Curve", legend = false)