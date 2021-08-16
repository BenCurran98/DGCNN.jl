using DGCNN

using Flux
using ParameterSchedulers

using Flux.Losses
using ParameterSchedulers: Stateful

η = 0.001
β1 = 0.9
β2 = 0.999

opt = ADAM(η, (β1, β2))

loss = crossentropy

epochs = 20

η_min = 1e-3
scheduler = Stateful(Cos(λ0 = η_min, λ1 = η, period = epochs))

model_dir = joinpath(@__DIR__, "model_store")

num_neigh = 30
num_classes = 5

model = Dgcnn_classifier(num_neigh, num_classes)

num_box_samples = 3
box_size = 30
class_min = 100
classes = [1, 2, 3, 4, 5]
num_points = 5000
sample_points = 100_000
training_batch_size = test_batch_size = 8
data_dir = "/media/ben/T7 Touch/InnovationConference/Datasets/QualityTraining"
training_batches, test_batches = get_data(data_dir; 
                                            sample_points = sample_points, 
                                            test_prop = 0.2,
                                            num_box_samples = num_box_samples,
                                            box_size = box_size,
                                            class_min = class_min,
                                            classes = classes,
                                            training_batch_size = training_batch_size,
                                            test_batch_size = test_batch_size)

train!(model, training_batches, test_batches, loss, opt, epochs; scheduler = scheduler, model_dir = model_dir)