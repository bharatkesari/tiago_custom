(define (problem recycle) (:domain recycle_bot)
(:objects 
    doorway_1 - doorway
    room_1 room_2 - room
    aruco_cube_1 - aruco_cube
    can_1 - can
    bin_1 - bin
    none - none
    robot_1 - robot
)

(:init
    ; always true states
    (connect room_1 room_2 doorway_1)
    (connect room_2 room_1 doorway_1)

    ; variable states
    (at room_1 robot_1)
    (at room_1 can_1)
    (at room_1 aruco_cube_1)
    (at room_2 bin_1)
    (facing none)
    (hold none)
    (at room_1 doorway_1)
    (at room_2 doorway_1)
)

(:goal (and
    ; (contain trash_1 bin_1)
    ; (facing trash_1)
    ; (at room_2 robot_1)
    ; (hold trash_1)
    ; (facing bin_1)
    (contain aruco_cube_1 bin_1)
    ; (facing doorway_1)
    )
)
)