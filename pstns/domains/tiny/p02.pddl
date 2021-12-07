(define(problem p01)
(:domain test-domain)
(:objects
  t01 t02 - token
  p01 p02 - place
)
(:init
  (not-hosted t01)
  (not-hosted t02)
  (at 10 (available p01))
  (at 40 (available p02))
  (deadline-open)
  (at 50 (not (deadline-open)))
)
(:goal (and
  (hosted t01)
  (hosted t02)
)))
