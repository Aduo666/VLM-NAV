from __future__ import annotations

from typing import Protocol

from .contracts import BeliefSnapshot, GeometryHypothesis, ObservedGeometry, RealObservation


class RealGeometryIntegrator(Protocol):
    def integrate(self, current: ObservedGeometry | None, observation: RealObservation) -> ObservedGeometry: ...


class Belief:
    def __init__(self, *, integrator: RealGeometryIntegrator | None = None,
                 hypotheses: tuple[GeometryHypothesis, ...] = ()):
        self._integrator = integrator
        self._snapshot = BeliefSnapshot(0, (), hypotheses=hypotheses)

    def snapshot(self) -> BeliefSnapshot:
        return self._snapshot

    def update(self, observation: RealObservation) -> BeliefSnapshot:
        if not isinstance(observation, RealObservation):
            raise TypeError("Belief.update 只接受 RealObservation")
        previous = self._snapshot
        if any(value.observation_id == observation.observation_id for value in previous.observations):
            raise ValueError("真实观测不能重复融合")
        if previous.observations and observation.frame.timestamp_sec < previous.observations[-1].frame.timestamp_sec:
            raise ValueError("真实观测需要按采集时间更新")
        geometry = previous.observed_geometry
        if self._integrator is not None:
            geometry = self._integrator.integrate(geometry, observation)
            if not isinstance(geometry, ObservedGeometry):
                raise TypeError("真实观测融合器只允许返回 ObservedGeometry")
        updated = BeliefSnapshot(previous.version + 1, previous.observations + (observation,),
                                geometry, previous.hypotheses)
        self._snapshot = updated
        return updated
