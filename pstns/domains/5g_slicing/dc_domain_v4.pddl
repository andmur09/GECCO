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


(define (domain dcwl_domain)

    (:requirements :strips :typing :equality :fluents :durative-actions :disjunctive-preconditions :duration-inequalities :timed-initial-literals :action-costs :negative-preconditions)
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
    ;v2+my detailed go through the modeling of the files with the assumption where only one component at one place
    (:types 
        service 
        pl sc esf - component
        spine-switch leaf-switch nodepod - place
        )

    (:functions
        (link-establishment-cost ?f-place ?t-place - place)
        (component-movement-cost ?f-transition-place ?t-transition-place - place)
        ; (component-placement-cost ?component - component)
        (place-cost ?place - place)
        (link-time)
        (place-time)
        (total-cost ?service - service)
    )


    (:predicates
        (latency ?place - place) 
        (throughput ?place - place)
        (packet-drop ?place - place)
        ; (protocol-in-use ?place - place ?protocol - protocol)
        (link-established ?service - service ?f-component ?t-component - component)
        (link-allowed ?service - service ?f-place ?t-place - place)
        (is-not-placed ?service - service ?component - component)
        (does-not-host ?place - place)
        (is-hosted-at ?service - service ?component - component ?at-place - place)
        (component-running ?service - service ?component - component ?at-place - place)
        (link-running ?service - service ?f-component - component ?t-component - component)
        (link-down ?service - service ?f-place - place ?t-place - place)
        (link-not-established ?service - service ?transition-component ?other-component - component)
	)

(:durative-action host-component 
            :parameters 
                (
                ?service - service
                ?component - component
                ?at-place - place 
                )
            :duration 
                (= ?duration 1) 
            :condition
	            (and 
                ;not sure if one place can host many components
                (at start(is-not-placed ?service ?component))
                (at start(does-not-host ?at-place))
                )
            :effect
	            (and 
	                (at end(is-hosted-at ?service ?component ?at-place))
                    (at start(not(is-not-placed ?service ?component))) 
                    (at start(not(does-not-host ?at-place))) 
                    (at start(increase(total-cost ?service) (place-cost ?at-place))) ;+component-placement-cost ?component
                )
    )

    (:durative-action establish-link
            :parameters 
                (?service - service ?f-component ?t-component - component ?f-place ?t-place - place)
        
            :duration 
                (= ?duration 2)
            :condition
	            (and 
	                (over all(link-allowed ?service ?f-place ?t-place)) ;;both places should be linked ;overall
                    (over all(is-hosted-at ?service ?f-component ?f-place));overall ;;service component1 should be on particular place
                    (over all(is-hosted-at ?service ?t-component ?t-place));overall ;;service component2 should be on particular place
                    (at start(component-running ?service ?f-component ?f-place))
                    (at start(component-running ?service ?t-component ?t-place))
                )
            :effect
	            (and 
                    (at end (link-established ?service ?f-component ?t-component)) 
                    ; (at end(not(link-down ?service ?f-place ?t-place)))
                    (at start(increase(total-cost ?service) (link-establishment-cost ?f-place ?t-place)))
                )
    )

    (:durative-action move-component
            :parameters 
                (?service - service 
                ?transition-component ?other-component - component 
                ?f-transition-place ?t-transition-place ?other-place - place)
        
            :duration 
                (= ?duration 2)
            :condition
	            (and 
	                (at start(is-hosted-at ?service ?transition-component ?f-transition-place))
                    (at start(is-hosted-at ?service ?other-component ?other-place))
                    ; (at start(not(link-established ?service ?transition-component ?other-component))) ;; ADL error
                    (at start(link-not-established ?service ?transition-component ?other-component))
                    (at start(does-not-host ?t-transition-place));;make sure t-transition-place is not hosting other components
                )
            :effect
	            (and 
                    (at start(not(does-not-host ?t-transition-place)))
                    (at end(is-hosted-at ?service ?transition-component ?t-transition-place))
                    (at start(not(is-hosted-at ?service ?transition-component ?f-transition-place)))
                    (at start(increase(total-cost ?service) (component-movement-cost ?f-transition-place ?t-transition-place)))
                    (at end(not(link-running ?service ?transition-component ?other-component)))
                    (at end(not(link-established ?service ?transition-component ?other-component)))
                )
    )

    (:durative-action keep-component-running
            :parameters 
                (?service - service
                ?component - component
                ?at-place - place               
                )
        
            :duration 
                (= ?duration place-time) 
        
            :condition
	            (and 
                (over all(is-hosted-at ?service ?component ?at-place)) 
                )
            :effect
	            (and 
                (at start(increase(total-cost ?service) place-time));;total-cost changed
                (at end(component-running ?service ?component ?at-place))
                )
    )

    (:durative-action keep-link-running
            :parameters 
                ( 
                ?service - service
                ?f-component ?t-component - component
                ?f-place ?t-place - place 
                )
        
            :duration 
                (= ?duration link-time) 

            :condition
	            (and 
                (over all(is-hosted-at ?service ?f-component ?f-place))
                (over all(is-hosted-at ?service ?t-component ?t-place))
                (over all(link-established ?service ?f-component ?t-component))
                (over all(link-allowed ?service ?f-place ?t-place))
                )
            :effect
	            (and 
                (at start(increase(total-cost ?service) link-time))
                (at end(link-running ?service ?f-component ?t-component))
                )
    )

    (:durative-action tear-down-link
            :parameters 
                (
                ?service - service
                ?transition-component ?other-component - component
                ?transition-place ?other-place - place 
                )
        
            :duration 
                (= ?duration 1)
        
            :condition
	            (and 
                (at start(is-hosted-at ?service ?transition-component ?transition-place))         
                (at start(is-hosted-at ?service ?other-component ?other-place))   
                (at start(link-established ?service ?transition-component ?other-component)) 
                )
            :effect
	            (and 
	            ; (at start(not(link-established ?service ?transition-component ?other-component)))
                (at start(link-not-established ?service ?transition-component ?other-component))
                )
    )
    
)

