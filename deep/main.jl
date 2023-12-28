"""
  Deep Q-Learning algorithm for solving the Frozen Lake problem.

  Description:
    DQN algorithm differs from the vanilla Q-Learning in a way, that Q-matrix,
    responsible for evaluating the profit from each possible decision, is
    replaced with a neural network that maps states to the ActionQ-value
    pairs. As a result, the algorithm becomes more flexible and scales better.

  Features: Main & Ideal Networks, Memory Replay, Exploration Decay.
"""

using Flux
using Plots

import Random: rand

include("types.jl")
include("model.jl")
include("../shared/utils.jl")
include("../shared/video.jl")

# *----------------------------------------------------------------------------* World Definition

"World definition."
const W::World = [
  [None None None None]
  [None Hole None Hole]
  [None None None Hole]
  [Hole None None Goal]
]

"Number of states."
const N = length(W)
const (Nx, Ny) = size(W)

# *----------------------------------------------------------------------------* Methods

"Transform coordinates to state."
state(X::Int, Y::Int)::Int = (X - 1) * Nx + Y

"Transform state to coordinates."
cords(X::Int)::Tuple{Int,Int} = ((X - 1) ÷ Nx + 1, (X - 1) % Ny + 1)

"Calculate reward for a given tile."
reward(C::Tile)::Float32 = float(C)

"Calculate reward for a given state."
reward(X::Int)::Float32 = float(W[X])

"Calculate reward for a given coordinates."
reward(X::Int, Y::Int)::Float32 = float(W[X, Y])

"Return the agent to the starting position."
reset!(A::Agent) = (A.X, A.Y) = (1, 1)

# *----------------------------------------------------------------------------* Policies

"Random action choice."
choiceRandom()::Move = rand(Moves)

"Greedy action choice."
choiceGreedy(A::Agent)::Move = Move(argmax(A.Q(state(A.X, A.Y))))

"""Choice with exploration rate `ϵ`."""
choice(A::Agent, ϵ::Float32=0.0f0)::Move = (
  ϵ > rand()
  ? choiceRandom()
  : choiceGreedy(A)
)

# *----------------------------------------------------------------------------* Interaction

"Make move and get the reward."
function make(A::Agent, M::Move)::Tuple{Int,Int,Float32,Bool}
  S₀ = state(A.X, A.Y)

  if M == Up && A.Y != Ny
    A.Y += 1
  elseif M == Down && A.Y != 1
    A.Y -= 1
  elseif M == Left && A.X != 1
    A.X -= 1
  elseif M == Right && A.X != Nx
    A.X += 1
  else
    return S₀, S₀, reward(Wall), false
  end

  return S₀, state(A.X, A.Y), reward(A.X, A.Y), W[A.X, A.Y] in (Goal, Hole)
end

# *----------------------------------------------------------------------------* Training

function train!(A::Agent, epochs::Int=1000, maxSteps::Int=50, sync::Int=5, α::Float32=0.01f0, γ::Float32=0.95f0)
  Qᵢ = model(length(W), length(Moves), A.Q) # target (ideal) q-network

  ls::Vector{Float32} = Vector{Float32}()
  op = Flux.setup(Flux.ADAM(α), A.Q)
  ms = Vector{Memory}() # memories

  for e in 1:epochs
    printline()
    empty!(ms)
    reset!(A)

    # Exploring The Environment #----------------------------------------------#

    for _ in 1:maxSteps
      M = choice(A, decay(e))
      S₁, S₂, R, done = make(A, M)

      push!(ms, Memory(S₁, S₂, M, R))
      println("$M → $R")

      done && break
    end

    # Processing Experience #--------------------------------------------------#

    for m in ms
      # maximal expected Q-value for the next state
      y = m.R + γ * Flux.reduce(Flux.max, Qᵢ(m.S₂))

      l, gs = Flux.withgradient(A.Q) do Q
        # Q-value for previous state and action taken
        x = Q(m.S₁)[m.M]
        Flux.mse(x, y)
      end

      Flux.update!(op, A.Q, gs[1])
      push!(ls, l)
    end

    # Network Weights Synchronization #----------------------------------------#

    e % sync == 0 && Qᵢ << A.Q

  end

  return ls
end

# *----------------------------------------------------------------------------* Testing

"Test the agent in exploitation mode."
function test(A::Agent)
  steps::Steps = [(1, 1)]
  done::Bool = false

  println("Testing")
  printline()
  reset!(A)

  while !done
    M = choice(A)
    _, S, _, done = make(A, M)

    push!(steps, cords(S))
  end

  return steps
end

# *----------------------------------------------------------------------------* Initialization & Training

A = Agent(model(length(W), length(Moves)))

loss = train!(A)

# *----------------------------------------------------------------------------* Loss Visualization

plot(loss; label="loss", xlabel="epoch", ylabel="MSE")

# *----------------------------------------------------------------------------* Testing

steps = test(A)

save(W, steps; framerate=5, name="result")