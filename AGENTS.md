# Isaac Scenes — Agent Guide

## Project vision

This project is intended to become an **automatic adversarial data-generation
pipeline** for robot-learning tasks in Isaac Sim. The current pick-and-place
(PnP) task is the first toy example used to develop and validate the pipeline;
it must not become a hard-coded assumption in the architecture. Future tasks
may use different robots, objects, goals, sensors, and interaction patterns.

The pipeline has two complementary agents:

- A **task agent** attempts to complete the task and generate successful robot
  trajectories.
- A **naughty agent** acts as an adversary. It may modify the simulated world
  at arbitrary times—for example by moving an object or changing its pose—to
  expose brittle behavior and generate recovery trajectories.

The word "adversarial" is inspired by the generator/adversary relationship in
GANs. In this project it means producing purposeful disturbances during task
execution, not merely randomizing the initial scene. Empirically, these
disturbed and recovered trajectories are expected to improve downstream model
success rates compared with training only on undisturbed trajectories.

## Automatic task generation

"Automatic" means that an agent should be able to generate, run, diagnose, and
revise code for any task described to the pipeline. Structure the repository
to make agent-driven trial and error cheap and observable:

- Keep task definitions, reusable robot skills, execution policy, adversarial
  disturbances, evaluation, and data recording as separate components.
- Prefer small modules with explicit inputs, outputs, preconditions, success
  conditions, and failure results.
- Keep experiment parameters in configuration rather than scattering them
  through control code.
- Make individual skills and disturbances runnable and testable in isolation.
- Record enough state, decisions, skill results, and disturbance metadata to
  explain why a rollout succeeded or failed.
- Do not design shared interfaces around PnP-specific phases, a Franka robot,
  or a single cube.

## Atomic skills

The execution model is inspired by systems such as RoboTwin 2.0 and
InternData-V1. Robot behavior should be exposed as a collection of **atomic
skills** implemented against stable project APIs. Examples might include
locating an object, moving to a pose, grasping, verifying a grasp, placing,
releasing, or returning home.

An atomic skill should:

- Perform one bounded capability rather than an entire task.
- Declare its required observations and parameters.
- Expose clear running, success, failure, and cancellation outcomes.
- Report effects that can be verified from the world rather than assuming an
  issued command succeeded.
- Avoid choosing the next task-level skill itself.

The task agent or a generated task policy decides **which skill to execute and
when**. A state machine, behavior tree, planner, or learned policy may provide
that sequencing, but task sequencing must remain separate from skill
implementation.

## Adversarial disturbances and recovery

The naughty agent owns disturbance selection and world mutation. It should
describe what it changed and when for logging and replay, but it should not
reach into the task agent and assign a PnP state or otherwise prescribe the
recovery procedure.

The task agent must remain closed-loop: observe the world, verify the
preconditions and effects of its current skill, detect invalidated assumptions,
and select an appropriate recovery skill. Disturbance events may help with
instrumentation, but observed simulator state is the source of truth. This
separation allows the same naughty agent interface to challenge tasks that do
not use the current PnP state machine.

Treat recovery attempts as valuable data. Preserve the disturbance, the
pre-disturbance context, the task agent's subsequent decisions, skill failures,
replans, and the final outcome in the generated trajectory record.
