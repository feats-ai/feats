#!/usr/bin/python
from __future__ import print_function
import sys
import linuxcnc


def move(x,y,z, f):
    s = linuxcnc.stat()
    c = linuxcnc.command()

    def ok_for_mdi():
		s.poll()
		return not s.estop and s.enabled and (s.homed.count(1) == s.joints) and (s.interp_state == linuxcnc.INTERP_IDLE)

    if ok_for_mdi():
        c.mode(linuxcnc.MODE_MDI)
        c.wait_complete() # wait until mode switch executed
        c.mdi("G1 G54 X{} Y{} Z{} f{}".format(x,y,z,f))
        c.wait_complete()
        s.poll()
        while s.interp_state != 1:
			s.poll()
        print("MDI command executed")
        return True

	return False
