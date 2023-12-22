using Flux

# *----------------------------------------------------------------------------* Weight Manipulation

"Copy one model's weights to another."
copyWeights!(source::Chain, target::Chain) = Flux.loadparams!(target, Flux.params(source))

"Copy one model's weights to another."
Base.:<<(target::Chain, source::Chain) = copyWeights!(source, target)

# *----------------------------------------------------------------------------* Model Construction

"Create a model."
function model(states::Int, actions::Int)::Chain
  return Flux.Chain(
    x -> Flux.onehot(x, 1:states),
    Flux.Dense(states => 10, init=Flux.kaiming_normal, Flux.relu),
    Flux.Dense(10 => actions, init=Flux.kaiming_normal),
    Flux.softmax
  )
end

"Create a model with weights taken from another model."
function model(states::Int, actions::Int, weightsFrom::Chain)::Chain
  result::Chain = model(states, actions)
  copyWeights!(weightsFrom, result)
  return result
end
