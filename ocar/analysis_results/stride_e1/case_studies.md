# STRIDE Detector Case Studies

## Case 1: Correctly Detected Stuck State

### tid=f01ca75b, ckpt=100, step=45

- step 43: S_θ=2.40 ΔS=-0.331 H_a=2.97 stuck=True 
  - obs: Nothing happens.
  - act: go to dresser 1

- step 44: S_θ=2.39 ΔS=-0.392 H_a=2.73 stuck=True 
  - obs: Nothing happens.
  - act: inventory

- step 45: S_θ=2.56 ΔS=-0.360 H_a=1.30 stuck=True **★ TRIGGER**
  - obs: Nothing happens.
  - act: examine sidetable 1

- step 46: S_θ=2.49 ΔS=-0.405 H_a=3.66 stuck=True 
  - obs: On the sidetable 1, you see nothing.
  - act: examine sofa 1

- step 47: S_θ=2.42 ΔS=-0.375 H_a=2.47 stuck=True 
  - obs: Nothing happens.
  - act: examine armchair 1 (exploration removes prejudiced conclusions)

### tid=f01ca75b, ckpt=100, step=46

- step 44: S_θ=2.39 ΔS=-0.392 H_a=2.73 stuck=True 
  - obs: Nothing happens.
  - act: inventory

- step 45: S_θ=2.56 ΔS=-0.360 H_a=1.30 stuck=True 
  - obs: Nothing happens.
  - act: examine sidetable 1

- step 46: S_θ=2.49 ΔS=-0.405 H_a=3.66 stuck=True **★ TRIGGER**
  - obs: On the sidetable 1, you see nothing.
  - act: examine sofa 1

- step 47: S_θ=2.42 ΔS=-0.375 H_a=2.47 stuck=True 
  - obs: Nothing happens.
  - act: examine armchair 1 (exploration removes prejudiced conclusions)

- step 48: S_θ=2.42 ΔS=-0.370 H_a=2.34 stuck=False 
  - obs: Nothing happens.
  - act: <think> 1. The keychain is not in your inventory. 2. There was an invisi-bag tur


## Case 2: Correctly Guarded Exploration (NOT triggered)

### tid=c8081333, ckpt=120, step=3

- step 1: S_θ=2.91 ΔS=+0.165 EMA_ΔS=+0.123 useful=True 
  - obs: You turn on the desklamp 1.
  - act: examine sidetable 1

- step 2: S_θ=2.59 ΔS=+0.054 EMA_ΔS=+0.089 useful=True 
  - obs: On the sidetable 1, you see a desklamp 1.
  - act: go to sidetable 1

- step 3: S_θ=2.67 ΔS=-0.126 EMA_ΔS=-0.019 useful=True **★ GUARDED** (exploring)
  - obs: You arrive at sofa 1. On the sofa 1, you see a remotecontrol 1.
  - act: go to sidetable 1

- step 4: S_θ=2.18 ΔS=-0.133 EMA_ΔS=-0.076 useful=False 
  - obs: You arrive at coffeetable 1. On the coffeetable 1, you see a keychain 3, a newspaper 1, a statue 2, and a tissuebox 1.
  - act: go to coffeetable 1

- step 5: S_θ=1.97 ΔS=-0.107 EMA_ΔS=-0.091 useful=False 
  - obs: On the coffeetable 1, you see a keychain 3, a newspaper 1, a statue 2, and a tissuebox 1.
  - act: go to sidetable 1

### tid=3ae00a93, ckpt=120, step=31

- step 29: S_θ=2.75 ΔS=+0.089 EMA_ΔS=+0.009 useful=False 
  - obs: Nothing happens.
  - act: examine sidetable 1

- step 30: S_θ=2.74 ΔS=-0.028 EMA_ΔS=-0.010 useful=False 
  - obs: Nothing happens.
  - act: go to sofa 1

- step 31: S_θ=2.58 ΔS=-0.036 EMA_ΔS=-0.023 useful=True **★ GUARDED** (exploring)
  - obs: You arrive at sofa 1. On the sofa 1, you see a laptop 1, a remotecontrol 2, and a remotecontrol 1.
  - act: go to coffeetable 1

- step 32: S_θ=2.43 ΔS=-0.059 EMA_ΔS=-0.041 useful=False 
  - obs: You arrive at coffeetable 1. On the coffeetable 1, you see a creditcard 2, and a watch 1.
  - act: go to sidetable 1

- step 33: S_θ=2.43 ΔS=-0.022 EMA_ΔS=-0.031 useful=False 
  - obs: You turn on the desklamp 1.
  - act: examine sidetable 1
