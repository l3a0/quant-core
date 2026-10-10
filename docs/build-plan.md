# Build plan — ithildin-core

The premise in [docs/design.md](design.md) sets the order. Adding a function
moves nothing and is cheap. Editing one is a re-pin of every consumer and is
expensive. So work that adds outranks work that edits, and the only thing that
reverses that is a defect in something a consumer already computes.

## How work gets cut

A module joins this package when it has consumers in two repositories. That is
the whole rule, and [CLAUDE.md](../CLAUDE.md) carries what it means in
practice. A candidate that fails it gets the reason written on its issue, so
the next session inherits the answer instead of the question.

A deliverable here is one module, because a module is the unit a consumer
imports and the unit a version moves.

## What is here

| Module | What it leaves usable | Consumers at v0.1.0 |
| --- | --- | --- |
| `ithildincore.timeseries` | Least squares, the ADF t-statistic at a fixed lag, the Ornstein-Uhlenbeck half-life, and the MacKinnon critical values | Both repositories |
| `ithildincore.stats` | The ordinary and Newey-West t-statistics of a series' mean, reported together | `trading-strategies` today, `quantitative-trading` at its first significance claim |

**Test surface.** Both modules are pure functions of their arguments, so every
rule is executable with no network and no fixture file. Three kinds of test
carry them.

1. **Closed-form cases.** A fit of `y = 2x + 1` returns exactly that, and a
   decay of `z_{t+1} = 0.5 z_t` has half-life `ln(2)/0.5`. These hold the
   arithmetic against an answer known in advance rather than against a
   previous run.
2. **Threshold properties.** The synthetic ADF cases assert the verdict the
   statistic implies rather than a seeded value that would say nothing if it
   moved.
3. **Pinned summaries.** Every field of one seeded `newey_west_summary` is
   held to six decimals. This is the guard the premise asks for: a change to
   the degrees of freedom, the Bartlett weights or the lag rule fails here
   rather than surfacing later as a consumer's number that quietly moved. No
   seeded series reaches the variance floor, so a change to the floor fails
   the floor test named below instead.

Three tests exist to hold claims the prose makes.

1. `TestTheFixedLagIsLoadBearing` executes the `timeseries` docstring's
   warning that a data-chosen lag can reverse a verdict, on a series where it
   does.
2. `test_the_variance_floor_is_not_reached_by_ordinary_data` is a smaller run
   of the random search the `stats` docstring cites.
3. `test_the_variance_floor_is_reached_from_23_observations` executes the
   `stats` docstring's claim that a series built for it reaches the floor from
   23 observations upward.

## The order

Nothing is queued. Both modules that meet the bar are here, and the next
addition waits for a second repository to need something a first one already
has.

That is a deliberate stopping point rather than an empty backlog. A package
that grows ahead of its consumers accumulates modules with one user, which is
the thing the bar exists to prevent.

## Dependencies

None between the modules. `ithildincore.stats` imports only `numpy` and
`ithildincore.timeseries` imports only `numpy` and `statsmodels`, so neither can
break the other and a consumer can import one without pulling the other's
dependencies.

That is a property to keep rather than a coincidence. A future module that
wants to build on `timeseries` should say so at the top of its own file, the
way `timeseries` says it imports nothing else here.
