"""
  Vanilla Q-Learning algorithm for solving the Frozen Lake problem.

  Algorithm:
    1. Initialize Q-Table.
    2. Choose an Action.
    3. Perform Action.
    4. Measure Reward.
    5. Update Q-Table.

  Terms:
    W - World
    S - State
    A - Agent
    R - Reward
    M - Move (Action)
    Q - Q-Table (Brain)
"""

# *----------------------------------------------------------------------------* Imports

import Random: rand

include("types.jl")
include("../shared/utils.jl")
include("../shared/video.jl")

# *----------------------------------------------------------------------------* Constants

"World definition."
const W::World = [
  [None None None None]
  [None Hole None Hole]
  [None None None Hole]
  [Hole None None Goal]
]

"World dimensions."
const (Nx::Int, Ny::Int) = size(W)

# *----------------------------------------------------------------------------* Methods

"Teleport agent to the starting position."
spawn(A::Agent) = (A.X, A.Y) = (1, 1)

"Calculate reward for a given tile."
reward(C::Tile)::Float32 = float(C)

"Calculate reward for a given location."
reward(X::Int, Y::Int)::Float32 = float(W[X, Y])

# *----------------------------------------------------------------------------* Policies

"Random action choice."
choiceRandom()::Move = rand(Moves)

"Greedy action choice."
choiceGreedy(A::Agent)::Move = Move(argmax(A.Q[A.X, A.Y, :]))

"Weighted random action choice."
# choiceWeight(A::Agent)::Move = sample(Moves, Weights(A.Q[A.X, A.Y, :])) # via StatsBase

"""Choice with exploration rate `ϵ`."""
choice(A::Agent, ϵ::Float32)::Move = (
  ϵ > rand() || iszero(A.Q[A.X, A.Y, :])
  ? choiceRandom()
  : choiceGreedy(A)
)

# *----------------------------------------------------------------------------* Interaction

"Make a move and get the reward."
make(A::Agent, M::Move)::Float32 = begin
  if M == Up && A.Y != Ny
    A.Y += 1
  elseif M == Down && A.Y != 1
    A.Y -= 1
  elseif M == Left && A.X != 1
    A.X -= 1
  elseif M == Right && A.X != Nx
    A.X += 1
  else
    return reward(Wall)
  end

  return reward(A.X, A.Y)
end

"Update Q-Table with `α` learning rate and `γ` discount factor."
learn(A::Agent, M::Move, X₀::Int, Y₀::Int, R::Float32; α::Float32=0.7f0, γ::Float32=0.95f0) =
  A.Q[X₀, Y₀, M] += α * (R + γ * maximum(A.Q[A.X, A.Y, :]) - A.Q[X₀, Y₀, M])

"Perform a step of action-choice, action-making, and learning."
function step(A::Agent, ϵ::Float32=0.0f0)::Tuple{Bool,Int,Int}
  X₀, Y₀ = A.X, A.Y

  M = choice(A, ϵ)

  R = make(A, M)

  println("$M → $R")

  learn(A, M, X₀, Y₀, R)

  return W[A.X, A.Y] in (Goal, Hole), A.X, A.Y
end

# *----------------------------------------------------------------------------* Helper Methods

"Train the agent for `episodes` episodes."
function train(A::Agent, episodes::Int)
  done::Bool = false
  steps::Steps = []
  x::Int = 1
  y::Int = 1

  for e in 1:episodes
    println("Episode: $e")
    printline()
    spawn(A)

    push!(steps, (1, 1))

    done = false

    while !done
      done, x, y = step(A, decay(e))
      push!(steps, (x, y))
    end

    printline()
    printspace()
  end

  return steps
end

"Test the agent in exploitation mode."
function test(A::Agent)
  steps::Steps = [(1, 1)]
  done::Bool = false

  println("Testing")
  printline()
  spawn(A)

  while !done
    done, x, y = step(A)
    push!(steps, (x, y))
  end

  return steps
end

# *----------------------------------------------------------------------------* Main

A::Agent = Agent(Nx, Ny, 4)

Hs = train(A, 1000)

H = test(A)

# *----------------------------------------------------------------------------* Video

save(W, Hs; framerate=25, name="training")

save(W, H; framerate=5, name="result")
