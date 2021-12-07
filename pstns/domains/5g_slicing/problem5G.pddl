(define (problem testslicing)
(:domain slicing)
(:objects 
    c0 - component
    dummy-pod p0 p1 - pod
)
(:init
    (= (component-run-time c0) 0)
    (= (operational-cost) 0)

    (running-on-pod c0 dummy-pod)
    (pod-free p0) (pod-real p0)
    (pod-free p1) (pod-real p1)

    (vm-not-initialised p0)
    (vm-not-initialised p1)

    (vm-configured dummy-pod)
    (vm-not-configured p0)
    (vm-not-configured p1)

    (= (time-to-connect dummy-pod p0) 1)
    (= (time-to-connect dummy-pod p1) 1)
    (= (time-to-connect p0 p1) 2)
    (= (time-to-connect p1 p0) 2)

    (= (vm-init-cost dummy-pod) 0)
    (= (vm-config-cost dummy-pod) 0)
    (= (vm-running-cost dummy-pod) 0)
    (= (vm-teardown-cost dummy-pod) 0)

    (= (vm-init-cost p0) 1)
    (= (vm-config-cost p0) 1)
    (= (vm-running-cost p0) 1)
    (= (vm-teardown-cost p0) 1)

    (= (vm-init-cost p1) 1)
    (= (vm-config-cost p1) 1)
    (= (vm-running-cost p1) 1)
    (= (vm-teardown-cost p1) 1)


    (= (running-cost p0) 1)
    (= (running-cost p1) 1)

    ;;(at 25 (violation p0))
    ;;(at 35 (violation p1))
    ;;(at 30 (not (violation p0)))
)  
(:goal (and
    ;;(running-on-pod c0 p0)
    ;;(>= (component-run-time c0) 10)
    ;;(>= (component-run-time c0) 30)
    (has-run p0)
    (has-run p1)
))

)