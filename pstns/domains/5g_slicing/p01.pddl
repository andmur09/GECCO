(define (problem fixfuse)
    (:domain slicing)
    
    (:objects    
        dc1 - node
        spine1 spine2 - node
        dummy-node node1 node2 - node
        dummy-port portA portB portC portD portE portF - port
        app1 - app 
    )
    
    (:init
        (= (total-cost) 0)
    
        (vm-not-initialised node1)
        (vm-not-configured node1)
        (vm-not-running node1)
        
        (vm-not-initialised node2)
        (vm-not-configured node2)
        (vm-not-running node2)
        
        (= (vm-running-cost node1) 2)
        (= (vm-init-cost node1) 10)
        (= (vm-config-cost node1) 10)
        (= (vm-teardown-cost node1) 10)
        
        (= (vm-running-cost node2) 2)
        (= (vm-init-cost node2) 10)
        (= (vm-config-cost node2) 10)
        (= (vm-teardown-cost node2) 10)
        
        (assigned app1 dummy-port)
        (= (app-total-time app1) 0)
        (= (required-latency app1) 50)
        (app-not-running app1)
        
        ;; tree
        (child node1 portA)
        (child node1 portB)
        (child node1 portC)
        (child node2 portD)
        (child node2 portE)
        (child node2 portF)
        (vm-configured dummy-node)
        (vm-running dummy-node)
        
        (= (latency node1) 10)
        (= (latency node2) 10)
        
        (at 150 (= (latency node2) 250))
        (at 250 (= (latency node2) 10))
        (at 550 (= (latency node1) 250))
        (at 675 (= (latency node1) 10))
        
        ;; fake port for allocation
        (= (time-to-connect dummy-port dc1) 0)
        (child dummy-node dummy-port)
        
        ;; connection costs/distances
        (= (time-to-connect node1 portA) 5)  (= (time-to-connect portA node1) 5)
        (= (time-to-connect spine1 portA) 5) (= (time-to-connect portA spine1) 5)
        (= (time-to-connect dc1 portA) 5)    (= (time-to-connect portA dc1) 5)
        (= (time-to-connect node1 portB) 5)  (= (time-to-connect portB node1) 5)
        (= (time-to-connect spine1 portB) 5) (= (time-to-connect portB spine1) 5)
        (= (time-to-connect dc1 portB) 5)    (= (time-to-connect portB dc1) 5)
        (= (time-to-connect node1 portC) 5)  (= (time-to-connect portC node1) 5)
        (= (time-to-connect spine1 portC) 5) (= (time-to-connect portC spine1) 5)
        (= (time-to-connect dc1 portC) 5)    (= (time-to-connect portC dc1) 5)
        (= (time-to-connect node2 portD) 5)  (= (time-to-connect portD node2) 5)
        (= (time-to-connect spine2 portD) 5) (= (time-to-connect portD spine2) 5)
        (= (time-to-connect dc1 portD) 5)    (= (time-to-connect portD dc1) 5)
        (= (time-to-connect node2 portE) 5)  (= (time-to-connect portE node2) 5)
        (= (time-to-connect spine2 portE) 5) (= (time-to-connect portE spine2) 5)
        (= (time-to-connect dc1 portE) 5)    (= (time-to-connect portE dc1) 5)
        (= (time-to-connect node2 portF) 5)  (= (time-to-connect portF node2) 5)
        (= (time-to-connect spine2 portF) 5) (= (time-to-connect portF spine2) 5)
        (= (time-to-connect dc1 portF) 5)    (= (time-to-connect portF dc1) 5)
        
    )
    
    (:goal (and 
        (>= (app-total-time app1) 29)
        ;(vm-configured node1)
        ;(vm-not-running node2)
        ;(assigned app1 portA)
    ))
    
    ;;(:metric minimize (total-cost))
)