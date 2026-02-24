import numpy as np 

def simulate_house_day(minuites = 1440):
    """
    using this to simulate realistic  multi room residential load profile 
    returns total load and per room load matriix
    """

    t = np.linspace(0 , 24 , minuites)
    # setting up base loads

    router = 0.05
    fridge = 0.15 + 0.05*np.sin(2* np.pi * t * 4) # added function for compressor cycles

    #bedroom
    bedroom = 0.1*((t>=6) & (t <= 8)) # simulating morning condition
    bedroom += 0.15 * ((t >= 20) & (t <= 23)) #simulating night condition


    # living room 

    hall = 0.2 * ((t >= 18 ) & (t <= 23)) #simulating evening usage 

    # kitchen base usage 
    kitchen = 0.3 * ((t >= 7) & (t <= 9))
    kitchen += 0.3 * ((t >= 19) & (t <= 21))

    #adding random spike events 

    spikes = np.zeros(minuites)

    for _ in range(np.random.randint(5, 10)):
        start = np.random.randint(0, minuites - 10)
        duration = np.random.randint(2, 8)
        power = np.random.uniform(0.8, 1.5) #setting up a high power device
        spikes[start:start + duration] += power 

    #combining loads 

    total = (
        router
        + fridge
        + bedroom
        + hall
        + kitchen
        + spikes
    )

    # setting up a small noise 

    total += 0.02* np.random.randn(minuites)
    total = np.clip(total, 0, None)

    #stacking per-room loads (excluding base)

    per_room = np.vstack([
        bedroom,
        hall,
        kitchen,
        spikes
    ]).T

    return total  , per_room





"""
here we are simulating base loads, time based and 


base (always on)

router 
fridge 
standby electronics

cyclical loads 

fridge compressor
cealing fan , 

water purifier pump 
shows on/off patterns

intermittent loads
tv
pc
lighting (heavy in the evenings )

spike loads(very critical for switching)

mixer grinder(1kw)
iron (1.2 kw)
hairdryer (1.5kw)
microwave (1 kw)

these loads are short yet aggressive bursts which trips inverters 

"""
    
