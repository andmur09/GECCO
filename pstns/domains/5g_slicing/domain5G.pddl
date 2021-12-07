(define (domain slicing)
(:requirements :fluents :durative-actions :duration-inequalities :adl :typing :time :timed-initial-literals)

(:types
  component
  pod
)

(:predicates
    ;; pods
    (pod-free ?p - pod)
    (pod-real ?p - pod)
    (violation ?p - pod)

    ;; service components
    (running-on-pod ?c - component ?p - pod)

    ;; vms
    (vm-initialised ?p - pod)
    (vm-not-initialised ?p - pod)
    (vm-configured ?p - pod)
    (vm-not-configured ?p - pod)


    (has-run ?p - pod)
)

(:functions
    (operational-cost)
    (running-cost ?p - pod)
    (component-run-time ?c - component)

    (time-to-connect ?p1 ?p2 - pod)

    (vm-init-cost ?p - pod)
    (vm-config-cost ?p - pod)
    (vm-running-cost ?p - pod)
    (vm-teardown-cost ?p - pod)
)

;; DAY-0 initialise a VM to migrate traffic
(:durative-action init-VM
    :parameters (?p - pod)
    :duration (= ?duration 1)
    :condition (and
        (at start (vm-not-initialised ?p))
        )
    :effect (and
        (at start (increase (operational-cost) (vm-init-cost ?p)))
        (at end (vm-initialised ?p))
        (at end (not (vm-not-initialised ?p)))
        )
)

;; DAY-1 configure a VM to migrate traffic
(:durative-action configure-VM
:parameters (?p - pod)
:duration (= ?duration 1)
:condition (and
    (at start (vm-not-configured ?p))
    (over all (vm-initialised ?p))
    )
:effect (and
    (at start (increase (operational-cost) (vm-config-cost ?p)))
    (at end (vm-configured ?p))
    (at start (not (vm-not-configured ?p)))
    )
)

;; DAY-1.5 Running VM
(:process run-VM
:parameters (?p - pod)
:precondition (and
    (vm-configured ?p)
    )
:effect(and
    (increase (operational-cost) (* #t (vm-running-cost ?p)))
    )
)

;; DAY-2 Migrate traffic
(:durative-action migrate-component
:parameters (?c - component ?p1 ?p2 - pod)
:duration (= ?duration (time-to-connect ?p1 ?p2))
:condition (and
    (at start (pod-free ?p2))
    (at start (running-on-pod ?c ?p1))
    (over all (vm-configured ?p1))
    (over all (vm-configured ?p2))
    (over all (pod-real ?p2))
    )
:effect (and
    (at start (not (pod-free ?p2)))
    (at start (not (running-on-pod ?c ?p1)))
    (at end (running-on-pod ?c ?p2))
    (at end (pod-free ?p1))
    (at end (increase (operational-cost) (time-to-connect ?p1 ?p2)))
    (at end (has-run ?p2))
    )
)

;; DAY-2 Running a component on a pod
(:process run-component-on-pod
:parameters (?c - component ?p - pod)
:precondition (and
    (running-on-pod ?c ?p)
    (pod-real ?p)
    )
:effect(and
    (increase (operational-cost) (* #t (running-cost ?p)))
    (increase (component-run-time ?c) (* #t 1))
    )
)

;; DAY-N Teardown VM
(:durative-action teardown-VM
:parameters (?p - pod)
:duration (= ?duration 1)
:condition (and
    (at start (vm-configured ?p))
    )
:effect (and
    (at start (increase (operational-cost) (vm-teardown-cost ?p)))
    (at end (vm-not-configured ?p))
    (at end (not (vm-configured ?p)))
    (at end (vm-not-initialised ?p))
    (at end (not (vm-initialised ?p)))
    )
)

;; SLA violation
(:event component-violation
:parameters (?c - component ?p - pod)
:precondition (and
    (running-on-pod ?c ?p)
    (violation ?p)
    )
:effect (and
    (not (running-on-pod ?c ?p))
    )
)

)
