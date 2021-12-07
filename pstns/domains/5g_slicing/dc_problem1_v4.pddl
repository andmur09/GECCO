; (c) Ericsson 2019 - All Rights Reserved
; No part of this material may be reproduced in any form
; without the written permission of the copyright owner.
; The contents are subject to revision without notice due
; to continued progress in methodology, design and manufacturing.
; Ericsson shall have no liability for any error or damage of any
; kind resulting from the use of these documents.
; Any unauthorized review, use, disclosure or distribution is
; expressly prohibited, and may result in severe civil and
; criminal penalties.
; Ericsson is the trademark or registered trademark of
; Telefonaktiebolaget LM Ericsson. All other trademarks mentioned
; herein are the property of their respective owners.



(define (problem dcwl_problem_1)

    ; ==============================================================================
    ; Datacenter workload placement planning for 5G slice assurance
    ; Using PDDL2.2. Tested on Optic-clp planner: https://nms.kcl.ac.uk/planning/software/optic.html 
    ; Initial state to be pulled from the KB
    ; Goal states arrive from the constraint solver.
    ; Author - Pooja Kashyap, Anusha Mujumdar(Trustworthy AI,RA AI)
    ; Reviewers - Sushanth David(Domain expert, BMAS), Swarup K. Mohalik, Marin Orlic, Aneta Vulgarakis(Trustworthy AI,RA AI)
    ; Maintainer - Anusha Mujumdar(Trustworthy AI,RA AI)
    ; Status - prototype
    ; ==============================================================================
    (:domain 
        dc_domain_v4
    )
    
    (:objects
        service1 service2 - service
        pl - pl
        sc - sc
        esf - esf
        leaf-switch1 - leaf-switch
        spine-switch1 - spine-switch
        node1pod1 node1pod2 node1pod3 node2pod4 node2pod5 node2pod6 node3pod7 - nodepod 
    )
    
    (:init
      
        (link-allowed service1 node1pod2 node1pod1)
        (link-allowed service1 node1pod3 node1pod1)
        (link-allowed service1 node2pod4 node1pod1)
        (link-allowed service1 node2pod5 node1pod1)
        (link-allowed service1 node2pod6 node1pod1)
        (link-allowed service1 node3pod7 node1pod1)
        (link-allowed service1 spine-switch1 node1pod1)
        (link-allowed service1 leaf-switch1 node1pod1)

        (link-allowed service1 node1pod1 node1pod2)
        (link-allowed service1 node1pod3 node1pod2)
        (link-allowed service1 node2pod4 node1pod2)
        (link-allowed service1 node2pod5 node1pod2)
        (link-allowed service1 node2pod6 node1pod2)
        (link-allowed service1 node3pod7 node1pod2) 
        (link-allowed service1 spine-switch1 node1pod2)
        (link-allowed service1 leaf-switch1 node1pod2)   

        (link-allowed service1 node1pod2 node1pod3)
        (link-allowed service1 node1pod1 node1pod3)
        (link-allowed service1 node2pod4 node1pod3)
        (link-allowed service1 node2pod5 node1pod3)
        (link-allowed service1 node2pod6 node1pod3)
        (link-allowed service1 node3pod7 node1pod3) 
        (link-allowed service1 spine-switch1 node1pod3)
        (link-allowed service1 leaf-switch1 node1pod3) 

        (link-allowed service1 node1pod2 node2pod4)
        (link-allowed service1 node1pod3 node2pod4)
        (link-allowed service1 node1pod1 node2pod4)
        (link-allowed service1 node2pod5 node2pod4)
        (link-allowed service1 node2pod6 node2pod4)
        (link-allowed service1 node3pod7 node2pod4)
        (link-allowed service1 spine-switch1 node2pod4)
        (link-allowed service1 leaf-switch1 node2pod4)

        (link-allowed service1 node1pod2 node2pod5)
        (link-allowed service1 node1pod3 node2pod5)
        (link-allowed service1 node2pod4 node2pod5)
        (link-allowed service1 node1pod1 node2pod5)
        (link-allowed service1 node2pod6 node2pod5)
        (link-allowed service1 node3pod7 node2pod5)
        (link-allowed service1 spine-switch1 node2pod5)
        (link-allowed service1 leaf-switch1 node2pod5)

        (link-allowed service1 node1pod2 node2pod6)
        (link-allowed service1 node1pod3 node2pod6)
        (link-allowed service1 node2pod4 node2pod6)
        (link-allowed service1 node2pod5 node2pod6)
        (link-allowed service1 node1pod1 node2pod6)
        (link-allowed service1 node3pod7 node2pod6)
        (link-allowed service1 spine-switch1 node2pod6)
        (link-allowed service1 leaf-switch1 node2pod6)

        (link-allowed service1 node1pod2 node3pod7)
        (link-allowed service1 node1pod3 node3pod7)
        (link-allowed service1 node2pod4 node3pod7)
        (link-allowed service1 node2pod5 node3pod7)
        (link-allowed service1 node2pod6 node3pod7)
        (link-allowed service1 node1pod1 node3pod7)
        (link-allowed service1 spine-switch1 node3pod7)
        (link-allowed service1 leaf-switch1 node3pod7)

        (link-allowed service1 node1pod2 spine-switch1)
        (link-allowed service1 node1pod3 spine-switch1)
        (link-allowed service1 node2pod4 spine-switch1)
        (link-allowed service1 node2pod5 spine-switch1)
        (link-allowed service1 node2pod6 spine-switch1)
        (link-allowed service1 node1pod1 spine-switch1)
        (link-allowed service1 node3pod7 spine-switch1)
        (link-allowed service1 leaf-switch1 spine-switch1)

        (link-allowed service1 node1pod2 leaf-switch1)
        (link-allowed service1 node1pod3 leaf-switch1)
        (link-allowed service1 node2pod4 leaf-switch1)
        (link-allowed service1 node2pod5 leaf-switch1)
        (link-allowed service1 node2pod6 leaf-switch1)
        (link-allowed service1 node1pod1 leaf-switch1)
        (link-allowed service1 node3pod7 leaf-switch1)
        (link-allowed service1 spine-switch1 leaf-switch1)
        
        ; (is-not-placed service1 pl)
        (is-not-placed service1 sc)
        (is-not-placed service1 esf)
  
        ; (=(component-placement-cost pl)2)
        ; (=(component-placement-cost sc)2)
        ; (=(component-placement-cost esf)2)


        ;;; for test purpose  ;;;
        (not(link-not-established service1 pl sc))
        (not(link-not-established service1 sc pl))
        (not(link-not-established service1 sc esf))
        (not(link-not-established service1 esf sc))
        (not(link-not-established service1 pl esf))
        (not(link-not-established service1 esf pl))

        (does-not-host node1pod1)
        ; (not(does-not-host node1pod2))
        (does-not-host node1pod3)
        (does-not-host node2pod4)
        (does-not-host node2pod5)
        (does-not-host node2pod6)
        (does-not-host node3pod7)
        (does-not-host leaf-switch1)
        (does-not-host spine-switch1)

        (=(place-cost node1pod1)111)
        (=(place-cost node1pod2)111)
        (=(place-cost node1pod3)113)
        (=(place-cost node2pod4)124)
        (=(place-cost node2pod5)125)
        (=(place-cost node2pod6)126)
        (=(place-cost node3pod7)137)
        (=(place-cost leaf-switch1)550)
        (=(place-cost spine-switch1)550)
        
        (=(component-movement-cost node1pod2 node1pod1)1)
        (=(component-movement-cost node1pod3 node1pod1)1)
        (=(component-movement-cost node2pod4 node1pod1)3)
        (=(component-movement-cost node2pod5 node1pod1)3)
        (=(component-movement-cost node2pod6 node1pod1)3)
        (=(component-movement-cost node3pod7 node1pod1)5)
        (=(component-movement-cost leaf-switch1 node1pod1)7)
        (=(component-movement-cost spine-switch1 node1pod1)7)

        (=(component-movement-cost node1pod1 node1pod2)1)
        (=(component-movement-cost node1pod3 node1pod2)1)
        (=(component-movement-cost node2pod4 node1pod2)3)
        (=(component-movement-cost node2pod5 node1pod2)3)
        (=(component-movement-cost node2pod6 node1pod2)3)
        (=(component-movement-cost node3pod7 node1pod2)5)  
        (=(component-movement-cost leaf-switch1 node1pod2)7)
        (=(component-movement-cost spine-switch1 node1pod2)7)  

        (=(component-movement-cost node1pod2 node1pod3)1)
        (=(component-movement-cost node1pod1 node1pod3)1)
        (=(component-movement-cost node2pod4 node1pod3)3)
        (=(component-movement-cost node2pod5 node1pod3)3)
        (=(component-movement-cost node2pod6 node1pod3)3)
        (=(component-movement-cost node3pod7 node1pod3)5)
        (=(component-movement-cost leaf-switch1 node1pod3)7)
        (=(component-movement-cost spine-switch1 node1pod3)7)

        (=(component-movement-cost node1pod2 node2pod4)3)
        (=(component-movement-cost node1pod3 node2pod4)3)
        (=(component-movement-cost node1pod1 node2pod4)3)
        (=(component-movement-cost node2pod5 node2pod4)1)
        (=(component-movement-cost node2pod6 node2pod4)1)
        (=(component-movement-cost node3pod7 node2pod4)3)
        (=(component-movement-cost leaf-switch1 node2pod4)7)
        (=(component-movement-cost spine-switch1 node2pod4)7)

        (=(component-movement-cost node1pod2 node2pod5)3)
        (=(component-movement-cost node1pod3 node2pod5)3)
        (=(component-movement-cost node2pod4 node2pod5)1)
        (=(component-movement-cost node1pod1 node2pod5)3)
        (=(component-movement-cost node2pod6 node2pod5)1)
        (=(component-movement-cost node3pod7 node2pod5)3)
        (=(component-movement-cost leaf-switch1 node2pod5)7)
        (=(component-movement-cost spine-switch1 node2pod5)7)

        (=(component-movement-cost node1pod2 node2pod6)3)
        (=(component-movement-cost node1pod3 node2pod6)3)
        (=(component-movement-cost node2pod4 node2pod6)1)
        (=(component-movement-cost node2pod5 node2pod6)1)
        (=(component-movement-cost node1pod1 node2pod6)3)
        (=(component-movement-cost node3pod7 node2pod6)3)
        (=(component-movement-cost leaf-switch1 node2pod6)7)
        (=(component-movement-cost spine-switch1 node2pod6)7)

        (=(component-movement-cost node1pod2 node3pod7)5)
        (=(component-movement-cost node1pod3 node3pod7)5)
        (=(component-movement-cost node2pod4 node3pod7)3)
        (=(component-movement-cost node2pod5 node3pod7)3)
        (=(component-movement-cost node2pod6 node3pod7)3)
        (=(component-movement-cost node1pod1 node3pod7)5)
        (=(component-movement-cost leaf-switch1 node3pod7)7)
        (=(component-movement-cost spine-switch1 node3pod7)7)

        (=(component-movement-cost node1pod2 leaf-switch1)7)
        (=(component-movement-cost node1pod3 leaf-switch1)7)
        (=(component-movement-cost node2pod4 leaf-switch1)7)
        (=(component-movement-cost node2pod5 leaf-switch1)7)
        (=(component-movement-cost node2pod6 leaf-switch1)7)
        (=(component-movement-cost node1pod1 leaf-switch1)7)
        (=(component-movement-cost node3pod7 leaf-switch1)7)
        (=(component-movement-cost spine-switch1 leaf-switch1)7)

        (=(component-movement-cost node1pod2 spine-switch1)7)
        (=(component-movement-cost node1pod3 spine-switch1)7)
        (=(component-movement-cost node2pod4 spine-switch1)7)
        (=(component-movement-cost node2pod5 spine-switch1)7)
        (=(component-movement-cost node2pod6 spine-switch1)7)
        (=(component-movement-cost node1pod1 spine-switch1)7)
        (=(component-movement-cost node3pod7 spine-switch1)7)
        (=(component-movement-cost leaf-switch1 spine-switch1)7)

        ;;;  link-establishment-cost   ;;;

        (=(link-establishment-cost node1pod2 node1pod1)10)
        (=(link-establishment-cost node1pod3 node1pod1)10)
        (=(link-establishment-cost node2pod4 node1pod1)10)
        (=(link-establishment-cost node2pod5 node1pod1)10)
        (=(link-establishment-cost node2pod6 node1pod1)10)
        (=(link-establishment-cost node3pod7 node1pod1)10)
        (=(link-establishment-cost spine-switch1 node1pod1)10)
        (=(link-establishment-cost leaf-switch1 node1pod1)10)
        ; (=(link-establishment-cost node1pod1 node1pod1)10)

        (=(link-establishment-cost node1pod1 node1pod2)10)
        (=(link-establishment-cost node1pod3 node1pod2)10)
        (=(link-establishment-cost node2pod4 node1pod2)10)
        (=(link-establishment-cost node2pod5 node1pod2)10)
        (=(link-establishment-cost node2pod6 node1pod2)10)
        (=(link-establishment-cost node3pod7 node1pod2)10)   
        (=(link-establishment-cost spine-switch1 node1pod2)10)
        (=(link-establishment-cost leaf-switch1 node1pod2)10)
        ; (=(link-establishment-cost node1pod2 node1pod2)10)

        (=(link-establishment-cost node1pod2 node1pod3)10)
        (=(link-establishment-cost node1pod1 node1pod3)10)
        (=(link-establishment-cost node2pod4 node1pod3)10)
        (=(link-establishment-cost node2pod5 node1pod3)10)
        (=(link-establishment-cost node2pod6 node1pod3)10)
        (=(link-establishment-cost node3pod7 node1pod3)10)
        (=(link-establishment-cost spine-switch1 node1pod3)10)
        (=(link-establishment-cost leaf-switch1 node1pod3)10)
        ; (=(link-establishment-cost node1pod3 node1pod3)10)

        (=(link-establishment-cost node1pod2 node2pod4)10)
        (=(link-establishment-cost node1pod3 node2pod4)10)
        (=(link-establishment-cost node1pod1 node2pod4)10)
        (=(link-establishment-cost node2pod5 node2pod4)10)
        (=(link-establishment-cost node2pod6 node2pod4)10)
        (=(link-establishment-cost node3pod7 node2pod4)10)
        (=(link-establishment-cost spine-switch1 node2pod4)10)
        (=(link-establishment-cost leaf-switch1 node2pod4)10)
        ; (=(link-establishment-cost node2pod4 node2pod4)10)

        (=(link-establishment-cost node1pod2 node2pod5)10)
        (=(link-establishment-cost node1pod3 node2pod5)10)
        (=(link-establishment-cost node2pod4 node2pod5)10)
        (=(link-establishment-cost node1pod1 node2pod5)10)
        (=(link-establishment-cost node2pod6 node2pod5)10)
        (=(link-establishment-cost node3pod7 node2pod5)10)
        (=(link-establishment-cost spine-switch1 node2pod5)10)
        (=(link-establishment-cost leaf-switch1 node2pod5)10)
        ; (=(link-establishment-cost node2pod5 node2pod5)10)

        (=(link-establishment-cost node1pod2 node2pod6)10)
        (=(link-establishment-cost node1pod3 node2pod6)10)
        (=(link-establishment-cost node2pod4 node2pod6)10)
        (=(link-establishment-cost node2pod5 node2pod6)10)
        (=(link-establishment-cost node1pod1 node2pod6)10)
        (=(link-establishment-cost node3pod7 node2pod6)10)
        (=(link-establishment-cost spine-switch1 node2pod6)10)
        (=(link-establishment-cost leaf-switch1 node2pod6)10)
        ; (=(link-establishment-cost node2pod6 node2pod6)10)

        (=(link-establishment-cost node1pod2 node3pod7)10)
        (=(link-establishment-cost node1pod3 node3pod7)10)
        (=(link-establishment-cost node2pod4 node3pod7)10)
        (=(link-establishment-cost node2pod5 node3pod7)10)
        (=(link-establishment-cost node2pod6 node3pod7)10)
        (=(link-establishment-cost node1pod1 node3pod7)100)
        (=(link-establishment-cost spine-switch1 node3pod7)10)
        (=(link-establishment-cost leaf-switch1 node3pod7)10)
        ; (=(link-establishment-cost node3pod7 node3pod7)10)

        (=(link-establishment-cost node1pod2 spine-switch1)10)
        (=(link-establishment-cost node1pod3 spine-switch1)10)
        (=(link-establishment-cost node2pod4 spine-switch1)10)
        (=(link-establishment-cost node2pod5 spine-switch1)10)
        (=(link-establishment-cost node2pod6 spine-switch1)10)
        (=(link-establishment-cost node1pod1 spine-switch1)10)
        (=(link-establishment-cost node3pod7 spine-switch1)10)
        (=(link-establishment-cost leaf-switch1 spine-switch1)10)
        ; (=(link-establishment-cost spine-switch1 spine-switch1)10)

        (=(link-establishment-cost node1pod2 leaf-switch1)10)
        (=(link-establishment-cost node1pod3 leaf-switch1)10)
        (=(link-establishment-cost node2pod4 leaf-switch1)10)
        (=(link-establishment-cost node2pod5 leaf-switch1)10)
        (=(link-establishment-cost node2pod6 leaf-switch1)10)
        (=(link-establishment-cost node1pod1 leaf-switch1)10)
        (=(link-establishment-cost node3pod7 leaf-switch1)10)
        (=(link-establishment-cost spine-switch1 leaf-switch1)10)
        ; (=(link-establishment-cost leaf-switch1 leaf-switch1)10)

        (=(link-time)10)
        (=(place-time)10)
        (is-hosted-at service1 pl node1pod2)
        (=(total-cost service1)0)

        ; (at 5(not(link-not-established service1 sc esf)))
        ; (at 4(not(link-not-established service1 pl sc)))
        ; (at 3(not(link-allowed service1 node1pod1 node1pod2)))
        ; (at 3(not(link-allowed service1 node1pod2 node1pod1)))
        ; (at 3(not(link-allowed service1 node1pod3 node1pod1)))
        ; (at 3(not(link-allowed service1 node1pod3 node1pod2)))
        ; (at 3(not(link-allowed service1 node1pod2 node1pod3)))
        ; (at 3(not(link-allowed service1 node1pod1 node1pod3)))
        ; (at 3(not(link-allowed service1 node1pod3 node1pod1)))
    )
    
    (:goal
        (and 
            ;; CASE1 
            (link-running service1 pl sc)
            (link-running service1 sc esf)
            (link-running service1 pl esf)
            (is-hosted-at service1 pl node1pod1)
        )
    )
    
    (:metric minimize (total-cost service1))
)
