#=
Q-Learning algorithm for solving the Frozen Lake problem.

Q-Learning Algorithm:
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
  M - Action (Move)
  Q - Q-Table (Memory)
=#

# *----------------------------------------------------------------------------* Imports

import Random: rand

include("lib/types.jl")
include("lib/visual.jl")

# *----------------------------------------------------------------------------* Constants

"World definition."
const W::Matrix{Tile} = [
  [None None None None]
  [None Hole None Hole]
  [None None None Hole]
  [Hole None None Goal]
]

"World dimensions."
const (Nx::Int, Ny::Int) = size(W)

"Minimum exploration rate."
const ϵₘᵢₙ = 0.05

# *----------------------------------------------------------------------------* Methods

# state(A::Agent)::State = State(
#   A.Y == Ny ? Wall : W[A.X, A.Y+1],
#   A.Y == 1  ? Wall : W[A.X, A.Y-1],
#   A.X == 1  ? Wall : W[A.X-1, A.Y],
#   A.X == Nx ? Wall : W[A.X+1, A.Y],
# )

"Teleport agent to the starting position."
spawn(A::Agent) = (A.X, A.Y) = (1, 1)

"Calculate reward for a given tile."
reward(C::Tile)::Float64 = float(C)

"Calculate reward for a given location."
reward(X::Int, Y::Int)::Float64 = float(W[X, Y])

# *----------------------------------------------------------------------------* Policies

"Random action choice."
choiceRandom()::Move = rand((Up, Down, Left, Right))

"Greedy action choice."
choiceGreedy(A::Agent)::Move = Move(argmax(A.Q[A.X, A.Y, :]))

"Weighted random action choice."
# choiceWeight(A::Agent)::Move = Move(argmax(A.Q[A.X, A.Y, :]))

"""Choice with exploration rate `ϵ`."""
choice(A::Agent, ϵ::Float64)::Move = (
  ϵ > rand() || iszero(A.Q[A.X, A.Y, :])
  ? choiceRandom()
  : choiceGreedy(A)
)

# *----------------------------------------------------------------------------* Interaction

"Make a move and get the reward."
make(A::Agent, M::Move)::Float64 = begin
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
learn(A::Agent, M::Move, X₀::Int, Y₀::Int, R::Float64; α::Float64=0.7, γ::Float64=0.95) =
  A.Q[X₀, Y₀, M] += α * (R + γ * maximum(A.Q[A.X, A.Y, :]) - A.Q[X₀, Y₀, M])

"Perform a step of action-choice, action-making, and learning."
function step(A::Agent, ϵ::Float64)::Bool
  X₀, Y₀ = A.X, A.Y

  # S::State = state(A)

  M = choice(A, ϵ)

  R = make(A, M)

  println("$M → $R")

  learn(A, M, X₀, Y₀, R)

  return W[A.X, A.Y] in (Goal, Hole)
end

# *----------------------------------------------------------------------------* Helper Methods

"Exponential decay function."
decay(λ::Float64, x::Int)::Float64 = (1.0 - ϵₘᵢₙ)exp(-λ * x) + ϵₘᵢₙ


printline() = println("-"^120)
printspace() = println("\n"^2)


"Train the agent for `episodes` episodes."
function train(A::Agent, episodes::Int)
  done::Bool = false

  for e in 1:episodes
    println("Episode: $e")
    printline()
    spawn(A)

    done = false

    while !done
      done = step(A, decay(1e-1, e))
    end

    printline()
    printspace()
  end
end


"Test the agent in exploitation mode."
function test(A::Agent)
  println("Testing")
  printline()
  spawn(A)

  done = false

  while !done
    done = step(A, 0.0)
  end
end

# *----------------------------------------------------------------------------* Main

A::Agent = Agent(Nx, Ny, 4)

train(A, 1000)

test(A)
