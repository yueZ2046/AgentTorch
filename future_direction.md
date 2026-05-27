# Urban Vitality Shenzhen - Future Direction

## 1. Purpose

This document defines the development direction for
`agent_torch.models.urban_vitality_shenzhen`. It complements `PROGRESS.md`:

- `PROGRESS.md` records completed experiments and current scores.
- This document states the current capability boundary, technical risks,
  target architecture, prioritized work, and acceptance criteria.

The long-term objective is to evolve the current Shenzhen vitality prototype
from observational fitting into a calibrated scenario-simulation capability
for comparing urban renewal interventions.

## 2. Current Status

The current module is an early prototype built on the differentiable
AgentTorch runtime. It predicts block-level LBS-present population for 48
time slots:

- 24 weekday hourly slots.
- 24 weekend hourly slots.

Current modeled entities and inputs:

| Item | Current implementation |
| --- | --- |
| Spatial units | 3,023 Shenzhen blocks |
| Agent representation | 4 demographic cohort agents per block, 12,092 total |
| Observed outcome | LBS-present population as a proxy for human-flow vitality |
| Input features | 70 built-environment, population, accessibility, and POI features |
| Spatial context | Neighbor80 graph and k-NN graph; k-NN is used for feature attention |
| Learnable behavior | cohort-by-time home probability and block attractiveness |
| Calibration | gradient-based fitting against observed LBS values |

The current result in `PROGRESS.md` shows material improvement over a naive
mean baseline. That establishes feasibility for block-level vitality
prediction, but it does not establish an intervention simulation system.

## 3. Capability Boundary

### 3.1 What the prototype currently does

- Loads Shenzhen block, population portrait, limited POI, and LBS data.
- Represents each block through four demographic cohorts.
- Learns temporal home/away tendencies and spatially contextualized block
  attractiveness.
- Aggregates cohort behavior into predicted block-level hourly human-flow
  vitality.
- Supports gradient flow through the modeled behavior and aggregation steps.

### 3.2 What the prototype does not yet do

The following should not be presented as implemented capabilities:

| Missing capability | Current limitation |
| --- | --- |
| Policy or renewal intervention simulation | No intervention variable is defined in the model state or transition rules. |
| Multi-round evolution | The model produces all 48 observation targets in one simulation step; it does not evolve city state over days or months. |
| LLM archetypes | Cohorts are learned parameter groups, not LLM-driven residents or archetypes. |
| OD-constrained mobility | Away population is not routed using observed origin-destination flows. |
| Resident interview or explanation | There is no conversational or memory-based agent layer. |
| Full urban vitality measurement | LBS-present population measures activity concentration, not economic, social, cultural, or experiential vitality in full. |
| Robust out-of-area generalization | Validation is currently based on held-out blocks, not strict spatial or policy-domain transfer tests. |

The appropriate present-tense description is:

> A Shenzhen block-level LBS human-flow vitality fitting prototype built on
> the AgentTorch differentiable simulation framework.

The appropriate future-tense ambition is:

> A calibrated scenario simulator capable of comparing candidate urban renewal
> interventions before implementation.

## 4. Current Technical Risks

### 4.1 Evaluation robustness is insufficient

The current best MAE is useful as an experiment result, but it is not yet a
stable model-quality claim. Required concerns include:

- A single random block split can overstate generalization.
- High-vitality outliers can dominate RMSE and materially affect conclusions.
- Repeated runs and alternative splits can vary because of initialization,
  split composition, and optimization behavior.
- Generated figures and prediction exports must be tied to an explicit
  experiment configuration and model version.

### 4.2 Mobility is only indirectly modeled

The current design learns a block attractiveness score and uses spatial
context, but it does not identify real flows between origins and destinations.
Without OD constraints:

- Attractive blocks may receive unrealistic inflows.
- Local routing may underperform because no observation anchors movement.
- Intervention effects on travel behavior cannot be interpreted reliably.

### 4.3 Important activity features are incomplete

Current usable POI enrichment covers only four categories:

- Medical services.
- Scenic locations.
- Automobile sales.
- Motorcycle services.

These do not fully express the activity types emphasized by the Shenzhen
vitality study, particularly food and beverage, shopping, life services,
leisure, public transport stations, and nighttime consumption.

### 4.4 Current state does not encode intervention semantics

The model can fit correlations in current conditions, but it cannot yet answer:

- What happens if retail or catering capacity is increased?
- What happens if an entrance, station connection, or public space is improved?
- What happens if nighttime events are introduced?
- Whether vitality is newly created or redistributed from adjacent blocks?

Solving this requires intervention variables with defensible causal or
behavioral interpretation, not only more predictive features.

## 5. Target Capability Architecture

The target system should evolve through four layers. These layers should be
built sequentially rather than claimed prematurely.

| Layer | Function | Required outputs |
| --- | --- | --- |
| L1 Observational prediction | Reproduce observed block and hourly activity patterns | Calibrated vitality baseline, uncertainty, diagnostics |
| L2 Mobility-constrained response | Model origins, destinations, and spatial redistribution | OD flow fit, inflow/outflow effects, displacement risk |
| L3 Intervention scenario simulation | Encode renewal actions as scenario parameters | Before/after and scheme-A/scheme-B comparisons |
| L4 Decision-facing explanation | Communicate results and limitations to users | Scenario reports, representative mechanisms, audit trail |

AgentTorch is most directly suited to L1-L3: tensorized population state,
spatial transitions, calibration, and parameterized scenario comparison.
Interview-style residents or narrative explanation should be considered a
separate later layer and must not be confused with the core quantitative
simulation.

## 6. Prioritized Roadmap

### Phase 0 - Make the current result reproducible

**Objective:** ensure every reported result can be regenerated and audited.

Tasks:

- Add a versioned experiment configuration recording:
  - random seed;
  - train/validation split strategy;
  - feature set;
  - graph configuration;
  - training epochs and optimizer parameters;
  - code commit or working-tree identifier.
- Generate predictions, metrics, and charts from one command and place them in
  a uniquely named run directory.
- Store MAE, RMSE, median absolute error, percentile errors, correlation, and
  performance by hour and vitality tier.
- Prevent stale figures from being presented as current-model results.
- Run multiple fixed seeds and report mean, standard deviation, best, and
  worst performance.

Acceptance criteria:

- A single command regenerates a complete result bundle.
- Every figure used externally references the run configuration.
- Results are reported across repeated seeds, not only from the best run.

### Phase 1 - Strengthen validation and baseline comparisons

**Objective:** determine whether the model captures transferable spatial
patterns rather than memorizing favorable splits.

Tasks:

- Implement spatially blocked validation by district or contiguous regions.
- Implement stratified validation across vitality tiers and identified
  vitality-circle types.
- Add stronger predictive baselines, for example:
  - regularized linear model;
  - gradient boosted trees;
  - spatial feature regression;
  - simple neural network without agent aggregation.
- Diagnose the highest-error blocks and peak-hour errors.
- Quantify whether the agent formulation improves prediction or primarily
  serves as an interpretable simulation structure.

Acceptance criteria:

- The prototype is compared against at least three non-agent baselines.
- Performance is available for random, spatial, and type-stratified splits.
- Outlier failure cases are documented with spatial and feature diagnostics.

### Phase 2 - Complete activity-relevant data inputs

**Objective:** align the model inputs with the vitality mechanisms described in
the broader Shenzhen study.

Tasks:

- Obtain complete geospatial POI files for:
  - catering and beverage;
  - shopping and retail;
  - life services;
  - sports and leisure;
  - metro and public transport access;
  - cultural and event-related facilities.
- Establish data provenance, update dates, coordinate systems, category
  mappings, and missing-data rules.
- Add activity-period features where supported, such as nighttime-serving or
  weekend-oriented facilities.
- Re-run ablation experiments to identify features that meaningfully improve
  prediction and interpretation.

Acceptance criteria:

- A feature dictionary links each modeled variable to a data source and
  vitality rationale.
- Core commercial, leisure, and transit activity variables are no longer
  missing from the usable POI set.
- Ablations show the contribution and limitations of added data groups.

### Phase 3 - Introduce OD-constrained mobility

**Objective:** move from attractiveness fitting toward interpretable movement
redistribution.

Tasks:

- Process raw arrival and departure LBS layers into block-level or grid-to-block
  OD observations where feasible.
- Define source population, destination choice, time-slot flow, and residual
  stay-home components explicitly.
- Replace or constrain unconstrained away-population allocation with observed
  flow-informed routing.
- Evaluate local versus citywide destination competition under OD supervision.
- Add conservation and capacity checks where appropriate.

Acceptance criteria:

- Predicted flows can be assessed against observed arrivals/departures.
- Destination changes are traceable to origin groups and time slots.
- Mobility behavior is stable enough to support limited counterfactual tests.

### Phase 4 - Encode urban renewal interventions

**Objective:** enable comparison of candidate update schemes, with explicitly
defined limits on causal interpretation.

Candidate scenario variables:

| Intervention category | Example model parameters |
| --- | --- |
| Functional mix | catering/retail/leisure facility additions or reductions |
| Public space | accessible open-space area, recreation capacity, event hosting |
| Transport access | station connection, walking accessibility, transfer cost |
| Temporal operation | evening opening time, weekend events, programmed activity |
| Affordability or inclusion | proxy parameters affecting cohort attractiveness |

Tasks:

- Define intervention variables separately from observed descriptive features.
- Specify how each intervention alters attractiveness, routing, timing, or
  cohort response.
- Build scenario APIs for baseline, intervention A, and intervention B.
- Report total vitality change together with redistribution and spillover.
- Require observational or pilot evidence before presenting simulated impacts
  as decision-grade estimates.

Acceptance criteria:

- At least one scenario can be run end to end with transparent parameter
  definitions.
- Outputs include temporal impacts, spatial impacts, cohort impacts, and
  spillover risks.
- Scenario results are labeled as counterfactual estimates with documented
  assumptions.

### Phase 5 - Multi-round evolution and decision workflow

**Objective:** extend from static scenario comparison toward an iterative
planning and monitoring loop.

Tasks:

- Redesign the runtime state so activity, attraction, intervention exposure,
  and observed feedback can evolve across multiple periods.
- Add event timelines for staged implementation, opening periods, holidays, or
  external shocks.
- Assimilate post-implementation observations to recalibrate parameters.
- Produce decision-facing reports with:
  - baseline versus scenario comparison;
  - uncertainty and boundary statements;
  - affected blocks and groups;
  - observed-versus-predicted monitoring after implementation.

Acceptance criteria:

- Multi-step state evolution is represented explicitly in code and tests.
- A simulated intervention can be evaluated before and after incoming
  observations are assimilated.
- Decision outputs retain a traceable link to inputs, assumptions, and model
  versions.

## 7. Deferred Capabilities

The following ideas may be valuable but should remain deferred until the
quantitative foundation is robust:

### LLM archetypes

LLM archetypes may later help express qualitative response differences among
population segments. They should not replace calibration against real
behavioral observations. Any implementation must define:

- what decisions are delegated to an LLM;
- how outputs are constrained and audited;
- how cost and stochasticity are controlled;
- how archetype behavior is validated against data.

### Resident interviews and narrative agents

Conversational agents may support stakeholder communication or scenario
storytelling. They must be labeled as explanatory interfaces, not direct
evidence of real resident intent.

### Automatic intervention optimization

Gradient-based optimization is attractive in principle, but only defensible
after intervention variables, objectives, constraints, and response functions
are explicitly modeled. Optimization should initially target bounded
continuous parameters within calibrated scenarios, not claim a globally
optimal urban renewal solution.

## 8. Development Principles

1. **Separate observed facts from simulated outcomes.** LBS data and measured
   urban attributes are observations; scenario predictions are model outputs.
2. **Prefer reproducible evidence over best-case metrics.** All public claims
   should reference a fixed run configuration and robust validation.
3. **Model interventions explicitly.** A feature correlation is not a policy
   effect unless the intervention mechanism is defined and validated.
4. **Retain spatial accountability.** Report who gains vitality, where it is
   lost, and whether changes represent creation or redistribution.
5. **Keep explanation subordinate to calibration.** Narrative or LLM layers
   should communicate quantitative outputs, not substitute for empirical
   grounding.
6. **Treat data legality and privacy as architecture requirements.** LBS and
   population data processing must use approved access, aggregation, and
   disclosure controls.

## 9. Recommended Immediate Work Items

The next implementation cycle should concentrate on foundation rather than
interface polish:

| Priority | Work item | Reason |
| --- | --- | --- |
| P0 | Add experiment run manifests and regenerate synchronized outputs | Current results cannot be responsibly reused without traceability. |
| P0 | Add repeated-seed and spatial-block validation | Model quality must be stable before scenario claims. |
| P1 | Diagnose outliers and peak-time error | Current errors can concentrate in high-impact blocks/time slots. |
| P1 | Acquire complete commercial/leisure/transit POI datasets | They directly represent the vitality mechanisms of interest. |
| P1 | Process arrival/departure data for OD supervision | Necessary for interpretable movement and spillover simulation. |
| P2 | Define the first intervention scenario schema | Establish the bridge from prediction to planning comparison. |
| P3 | Investigate LLM archetypes or interview interfaces | Only after quantitative layers are validated. |

## 10. Definition of Success

The project will have moved beyond an observational prototype when it can
demonstrate all of the following:

- Stable prediction performance under repeated and spatially meaningful
  validation.
- Complete and documented activity-relevant data inputs.
- Mobility outputs constrained and assessed using observed flow evidence.
- Explicitly parameterized renewal interventions.
- Scenario comparisons reporting uncertainty, spillover, and affected groups.
- A monitoring loop capable of recalibrating simulated effects with
  post-implementation observations.

Until then, the system should be described as a promising Shenzhen vitality
prediction and simulation research prototype, not as a completed urban
decision engine.
