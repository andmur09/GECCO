(define (domain test-domain)

    (:requirements
        :durative-actions
        :equality
        :negative-preconditions
        :numeric-fluents
        :object-fluents
        :typing
    )

    (:types
        token
        place
    )

    (:predicates
        (at ?t - token ?p - place)
        (hosted ?t - token)
        (not-hosted ?t - token)
        (available ?p - place)
        (deadline-open)
    )
    
    (:durative-action host-token
        :parameters (?t - token ?p - place)
        :duration (= ?duration 5)
        :condition (and 
            (at start (not-hosted ?t))
            (at start (available ?p))
            (over all (deadline-open))
            )
        :effect (and
            (at start (not (not-hosted ?t)))
            (at start (not (available ?p)))
            (at end (hosted ?t))
            (at end (at ?t ?p))
            )
    )
)