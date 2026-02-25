class HybridController:
    """
    Hybrid Solar–Battery–Grid Controller

    Controller does NOT select a single source.
    It decides:

    - allow_battery_discharge
    - force_grid
    - shed_load

    Solar is always used first in simulation.
    """

    def __init__(
        self,
        inverter_max_kw,
        soc_min=0.2,
        soc_reserve_night=0.3,
        spike_threshold_kw=3.0,
        spike_hold_steps=3,
    ):

        self.inverter_max_kw = inverter_max_kw
        self.soc_min = soc_min
        self.soc_reserve_night = soc_reserve_night
        self.spike_threshold_kw = spike_threshold_kw
        self.spike_hold_steps = spike_hold_steps

        self.previous_load = 0.0
        self.spike_timer = 0

    # -------------------------------------------------
    def decide(
        self,
        load_kw,
        battery_soc,
        grid_available=True,
        is_night=False,
    ):
        """
        Returns:
        {
            "allow_battery": bool,
            "force_grid": bool,
            "shed_load": bool
        }
        """

        decision = {
            "allow_battery": True,
            "force_grid": False,
            "shed_load": False,
        }

        # =================================================
        # 1️⃣ HARD SAFETY
        # =================================================

        if not grid_available:
            if load_kw > self.inverter_max_kw:
                decision["shed_load"] = True
                return decision

        if load_kw > self.inverter_max_kw:
            decision["force_grid"] = True
            decision["allow_battery"] = False
            return decision

        if battery_soc <= self.soc_min:
            decision["allow_battery"] = False
            decision["force_grid"] = True
            return decision

        if is_night and battery_soc <= self.soc_reserve_night:
            decision["allow_battery"] = False
            decision["force_grid"] = True
            return decision

        # =================================================
        # 2️⃣ SPIKE DETECTION
        # =================================================

        delta = load_kw - self.previous_load
        self.previous_load = load_kw

        if delta > self.spike_threshold_kw:
            self.spike_timer = self.spike_hold_steps

        if self.spike_timer > 0:
            self.spike_timer -= 1
            decision["force_grid"] = True
            decision["allow_battery"] = False

        return decision