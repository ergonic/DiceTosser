import ev3_dc as ev3
from thread_task import Sleep
def DiceItRirect(my_ev3):

    print(my_ev3.position)

    my_ev3.movement_plan = (
                my_ev3.move_to(-30, speed=100, ramp_up=100, ramp_down=100, brake=True) +
                #Sleep(0.05) +
                my_ev3.move_to(0, speed=100, ramp_up=100, ramp_down=100, brake=True) +
                Sleep(0.5) +
                my_ev3.stop_as_task(brake=False)
        )

    my_ev3.movement_plan.start(thread=False)
        #movement_plan.join()

    pass

my_ev3 = ev3.Motor(
        ev3.PORT_D,
        protocol=ev3.USB
    )

for _ in range(5):
    DiceItRirect(my_ev3)