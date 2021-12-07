;; QUESTIONS:
;; 1. Virtual machine runs for migration, or whole time? (we have only for migration)
;; 2. Virtual machine has to run on both source and destination during migration? (we assume both)
;; 3. Costs and durations for migrations both vary?
(define (domain slicing)

    (:requirements :strips :typing :fluents :durative-actions :timed-initial-literals :numeric-fluents)

    (:types
    
        ;; switches represent DC-gateway, spines, and leaves, nodes, and ports
        ;; they are arranged in a tree-structure.
        app switch - object
        gate spine node port - switch
    )

    (:predicates
    
        ;; apps
        (required-latency ?a - app)
        (assigned ?a - app ?p - port)
        (in-migration ?a - app)
        (app-running ?a - app)
        (app-not-running ?a - app)
        (just-assigned ?a - app ?p - port)

        ;; tree-structure of switches
        (child ?sp ?sc - switch)
        
        ;; VM predicates
        (vm-not-running ?n - node)
        (vm-running ?n - node)
        
        (vm-not-initialised ?n - node)
        (vm-initialised ?n - node)
        
        (vm-not-configured ?n - node)
        (vm-configured ?n - node)
        (vm-torn-down ?n - node)
    )

    (:functions
        (total-cost)
    
        (latency ?n - node)
        (required-latency ?a - app)
    
        (vm-init-cost ?n - node)
        (vm-config-cost ?n - node)
        (vm-teardown-cost ?n - node)
        (vm-running-cost ?n - node)
        
        (time-to-connect ?sa ?sb - switch)
        
        (app-total-time ?a - app)
    )

    ;; DAY-0 initialise a VM to migrate traffic
    (:durative-action init_VM
        :parameters (?n - node)
        :duration (= ?duration 2)
        :condition (and
            (at start (vm-not-initialised ?n))
            (over all (vm-running ?n))
            )
        :effect (and
            (at start (increase (total-cost) (vm-init-cost ?n)))
            (at end (vm-initialised ?n))
            (at end (not (vm-not-initialised ?n)))
            )
    )
    
    ;; DAY-1 configure a VM to migrate traffic
    (:durative-action configure-VM
        :parameters (?n - node)
        :duration (= ?duration 2)
        :condition (and
            (at start (vm-not-configured ?n))
            (over all (vm-initialised ?n))
            (over all (vm-running ?n))
            )
        :effect (and
            (at start (increase (total-cost) (vm-config-cost ?n)))
            (at end (vm-configured ?n))
            (at end (not (vm-not-configured ?n)))
            )
    )
    
    ;; DAY-2 Migrate traffic
    (:durative-action migrate-app
        :parameters (?a - app ?p1 ?p2 - port ?n1 ?n2 - node ?s - switch)
        :duration (= ?duration (+ (time-to-connect ?p1 ?s) (time-to-connect ?s ?p2)))
        :condition (and
            (at start (assigned ?a ?p1))
            (at start (child ?n1 ?p1))
            (at start (child ?n2 ?p2))
            (over all (vm-running ?n1))
            (over all (vm-configured ?n1))
            (over all (vm-running ?n2))
            (over all (vm-configured ?n2))
            )
        :effect (and
            (at end (increase (total-cost) (+ (time-to-connect ?p1 ?s) (time-to-connect ?s ?p2))))
            (at end (not (assigned ?a ?p1)))
            (at end (assigned ?a ?p2))
            (at end (just-assigned ?a ?p2))
            )
    )
    
    ;; DAY-N tear down a VM
    (:durative-action teardown-VM
        :parameters (?n - node)
        :duration (= ?duration 2)
        :condition (and
            (at start (vm-configured ?n))
            (at start (vm-running ?n))
            )
        :effect (and
            (at start (increase (total-cost) (vm-teardown-cost ?n)))
            (at start (vm-not-initialised ?n))
            (at start (not (vm-initialised ?n)))
            (at start (vm-not-configured ?n))
            ; (at start (not (vm-configured ?n)))
            (at start (vm-torn-down ?n))
            (at end (not (vm-torn-down ?n)))
            )
    )
    
    ;; models the running cost of the VM
    (:durative-action run-VM
        :parameters (?n - node)
        :duration (>= ?duration 20)
        :condition (and
            (at start (vm-not-running ?n))
            (at end (vm-torn-down ?n))
            )
        :effect (and
            (at start (vm-running ?n))
            (at start (not (vm-not-running ?n)))
            (at end (not (vm-running ?n)))
            (at end (vm-not-running ?n))
            (increase (total-cost) (* (vm-running-cost ?n) #t))
            )
    )
    
    ;; models the envelope of an app running
    (:durative-action run-app
        :parameters (?a - app ?p - port ?n - node)
        :duration (= ?duration 30)
        :condition (and
            (at start (app-not-running ?a))
            (over all (assigned ?a ?p))
            
            (over all (child ?n ?p))
            (over all (<= (latency ?n) (required-latency ?a)))
            
            (at start (in-migration ?a))
            ; (at end (in-migration ?a))
            )
        :effect (and
            (at start (not (app-not-running ?a)))
            (at start (app-running ?a))
            
            (at end (app-not-running ?a))
            (at end (not (app-running ?a)))
            
            (at end (increase (app-total-time ?a) ?duration))
            )
    )
    
    ;; bridges the run action between two ports
    (:durative-action run-clip
        :parameters (?a - app ?p1 ?p2 - port)
        :duration (= ?duration 0.2)
        :condition (and
            (at start (assigned ?a ?p1))
            (at end (just-assigned ?a ?p2))
            )
        :effect (and
            (at start (in-migration ?a))
            (at end (not (just-assigned ?a ?p2)))
            (at end (not (in-migration ?a)))
            )
    )
    
    ; ;; completes an app
    ; (:durative-action finish-clip
    ;     :parameters (?a - app)
    ;     :duration (= ?duration 0.2)
    ;     :condition (and
    ;         (at start (>= (app-total-time ?a) 300))
    ;         )
    ;     :effect (and
    ;         (at start (in-migration ?a))
    ;         (at end (not (in-migration ?a)))
    ;         )
    ; )
)