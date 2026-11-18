# metaworldplus — Scoring sections (curation aid)

## metaworldplus-assembly-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `assembly-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `assembly-v3` the check is: The centre of the nut's ring is within 2 cm of the target point horizontally (x and y) and lower than it: the ring is around the peg, below the target point at the peg's top. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-basketball-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `basketball-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `basketball-v3` the check is: The ball's centre is within 8 cm of the point that has the target point's x and y and the height z = 0.30 m, with the height difference counted double. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-bin-picking-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `bin-picking-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `bin-picking-v3` the check is: The object's centre is within 5 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-box-close-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `box-close-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `box-close-v3` the check is: The lid (measured at its handle) is less than 8 cm from the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-button-press-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `button-press-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `button-press-v3` the check is: The button's face is within 2 cm of the target point along y; only y is compared (the button is pushed in). ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-button-press-topdown-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `button-press-topdown-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `button-press-topdown-v3` the check is: The button's top is within 2.4 cm of the target point in height; only z is compared (the button is pushed down). ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-coffee-button-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `coffee-button-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `coffee-button-v3` the check is: The button's face is within 2 cm of the target point along y; only y is compared (the button is pushed in). ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-coffee-pull-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `coffee-pull-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `coffee-pull-v3` the check is: The mug's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-coffee-push-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `coffee-push-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `coffee-push-v3` the check is: The mug's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-dial-turn-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `dial-turn-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `dial-turn-v3` the check is: The dial's marker point (on its rim, 5 cm from the dial's axis, turning with the dial) is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-disassemble-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `disassemble-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `disassemble-v3` the check is: The nut is higher than the target point; only the height (z) is compared, measured at the nut's handle. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-door-close-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `door-close-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `door-close-v3` the check is: The door's handle is within 8 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-door-lock-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `door-lock-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `door-lock-v3` the check is: The lock's tip is within 2 cm of the target point in height; only z is compared. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-door-open-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `door-open-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `door-open-v3` the check is: The door's handle is within 8 cm of the target point along x; only x is compared. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-door-unlock-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `door-unlock-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `door-unlock-v3` the check is: The lock's tip is within 2 cm of the target point along x; only x is compared. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-drawer-close-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `drawer-close-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `drawer-close-v3` the check is: The drawer's handle is within 6.5 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-drawer-open-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `drawer-open-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `drawer-open-v3` the check is: The drawer's handle is within 3 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-faucet-close-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `faucet-close-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `faucet-close-v3` the check is: The end of the faucet's handle is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-faucet-open-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `faucet-open-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `faucet-open-v3` the check is: A point at the end of the faucet's handle (the handle's end shifted by -4 cm in x and +2 cm in z) is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-hand-insert-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `hand-insert-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `hand-insert-v3` the check is: The object's centre is within 5 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-handle-press-side-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `handle-press-side-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `handle-press-side-v3` the check is: The handle's grip is within 2 cm of the target point in height; only z is compared. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-handle-pull-side-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `handle-pull-side-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `handle-pull-side-v3` the check is: The handle's centre is within 8 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-lever-pull-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `lever-pull-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `lever-pull-v3` the check is: The lever has been turned to within 7.5 degrees (pi/24) of a quarter turn (90 degrees) from the position it starts in. Only that angle is compared; nothing is measured against the target point, which marks where the lever's end is after the turn. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-peg-insert-side-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `peg-insert-side-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `peg-insert-side-v3` the check is: The peg's head is within 7 cm of the target point, with the y and z differences counted double. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-peg-unplug-side-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `peg-unplug-side-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `peg-unplug-side-v3` the check is: The peg's end is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-pick-out-of-hole-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `pick-out-of-hole-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `pick-out-of-hole-v3` the check is: The object's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-pick-place-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `pick-place-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `pick-place-v3` the check is: The object's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-pick-place-wall-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `pick-place-wall-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `pick-place-wall-v3` the check is: The object's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-plate-slide-back-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `plate-slide-back-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `plate-slide-back-v3` the check is: The plate's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-plate-slide-back-side-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `plate-slide-back-side-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `plate-slide-back-side-v3` the check is: The plate's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-plate-slide-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `plate-slide-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `plate-slide-v3` the check is: The plate's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-plate-slide-side-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `plate-slide-side-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `plate-slide-side-v3` the check is: The plate's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-push-back-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `push-back-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `push-back-v3` the check is: The object's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-push-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `push-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `push-v3` the check is: The object's centre is within 5 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-push-wall-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `push-wall-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `push-wall-v3` the check is: The object's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-reach-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `reach-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `reach-v3` the check is: The point between the fingertips (`tcp`) is within 5 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-reach-wall-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `reach-wall-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `reach-wall-v3` the check is: The point between the fingertips (`tcp`) is within 5 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-shelf-place-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `shelf-place-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `shelf-place-v3` the check is: The object's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-soccer-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `soccer-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `soccer-v3` the check is: The ball's centre is within 7 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-stick-pull-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `stick-pull-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `stick-pull-v3` the check is: The thermos's handle (the loop the stick goes into) is within 12 cm of the target point AND the stick is still in the handle: the stick's far end is at or beyond the handle in x, within 4 cm of it in y and within 6 cm of it in z. A thermos that is at the target with the stick pulled out does not count. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-stick-push-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `stick-push-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `stick-push-v3` the check is: The thermos (the point 9 cm in +y from its handle) is within 12 cm of the target point AND the stick is still held and lifted: both finger pads press on the stick, the gripper is not fully shut, and the stick is more than 1 cm above the height it started at. A thermos that is at the target after the stick has been let go does not count. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-sweep-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `sweep-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `sweep-v3` the check is: The object's centre is within 5 cm of the target point. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-sweep-into-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `sweep-into-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `sweep-into-v3` the check is: The object's centre is within 5 cm of the target point horizontally (x and y); the height is not compared. ("The target point" is `s.state()["target_pos"]`.)


## metaworldplus-window-open-i00
The verifier replays your actions from the frozen start in a separate sandbox (once, in a fresh process; MuJoCo on the
CPU is bit-exact) and applies Meta-World's own success check for `window-open-v3` (`evaluate_state`) to the state **after
the last action**. A success reached earlier and then lost does not count: end the trajectory while the task is
achieved. `s.check()` computes exactly this check on the current state.

For `window-open-v3` the check is: The window's handle is within 5 cm of the target point along x; only x is compared. ("The target point" is `s.state()["target_pos"]`.)

