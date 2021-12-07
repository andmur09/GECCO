(define(problem p01)
(:domain test-domain)
(:objects
  t01 - token
  p01 - place
)
(:init
  (not-hosted t01)
  (deadline-open)
  (at 20 (not (deadline-open)))
  (at 10 (available p01))
)
(:goal (and
  (hosted t01)
)))
