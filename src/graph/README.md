# graph — MDP deterministico DoorKey

Costruisce (o carica da cache) il grafo completo dell'MDP `MiniGrid-DoorKey-{size}x{size}-v0` con accesso indicizzato O(1) e mappa ASCII per nodo.

* Ispirato a `bak/scripts/state_export2.py` per transizioni, stage e reward.
* Mappa per nodo replica `env/view_wrapper.py:151` (legenda sotto).

## Struttura

```
graph/
  mdp_graph.py   # build + render + save/load
  ...
cache in ../output/cache/: mdp_{size}x{size}_seed{SEED}.json  (path da src/paths.py)
```

Seed nel nome file (es. `mdp_8x8_seed42.json`).

## Uso CLI

```bash
python -m graph.mdp_graph --seed 42 --size 8          # build o cache
python -m graph.mdp_graph --seed 42 --size 8 --force  # rigenera
python -m graph.mdp_graph --seed 0 --size 6 --show 1  # 6x6
python -m graph.mdp_graph --seed 999 --size 16        # 16x16
```

## Uso Python

```python
from graph import load_or_build, render_map

mdp = load_or_build(seed=42, size=8)  # carica se esiste, altrimenti crea
# mdp = {seed, size, gamma, grid_info, nodes, index, adj, action_names}

# accesso indicizzato
state = (1, 5, 1, False, False)  # x,y,dir,has_key,door_open
nid = mdp["index"][state]
node = mdp["nodes"][nid]
print(node["x"], node["y"], node["stage"], node["is_terminal"])
print(node["map"])
print(node["transitions"][2])  # forward -> {next_id, next_state, reward, done}

# render arbitrario
print(render_map((2,2,0,True,False), mdp["grid_info"]))
```

### Nodo

```python
{
  "id": int,
  "state": (x,y,dir,has_key,door_open),
  "x": int, "y": int, "dir": int,          # espliciti richiesti
  "has_key": bool, "door_open": bool,
  "stage": "find_key"|"open_door"|"reach_goal",
  "is_terminal": bool,                     # (x,y)==goal
  "map": str,                              # ASCII con legenda
  "transitions": {
     0: {"next_id": int, "next_state": tuple, "reward": 0/1, "done": bool, "action_name": "left"},
     1: ..., 2: forward, 3: pickup, 5: toggle
  }
}
```

`reward = 1.0` solo entrando nel goal, `gamma=0.99` salvato a livello root per completezza. Transizioni deterministiche con self-loop su azione invalida. Nodo terminale = assorbente.

## Legenda mappa

```
+----+----+----+----+----+----+
|    |    |    | ▇  |    |    |
...
|L(D)|D(C)|    |    |    |    |  # L=con chiave, A=senza, K=key, G=goal
|    |    | ▇  |    |    | G  |
+----+----+----+----+----+----+

A(U/D/R/L)=agent, L(U/D/R/L)=con chiave, K=key, G=goal, ▇=wall
D(C)=door locked (alias D(L)), D(O)=door open
Stage: find_key / open_door / reach_goal
```

`x,y` del nodo coincidono con posizione `A()`/`L()` nella mappa. `D(C)` è alias di `D(L)` (closed=locked) per compatibilità con `view_wrapper.py:192` che usa `D(C)` vs `D(O)`.

## Cache

* `mdp_{size}x{size}_seed{seed}.json` — cache unica (human-readable, tuple→list, walls→sorted list; tipi nativi ricostruiti al load)

`load_or_build(..., force=True)` rigenera.

## Dettagli

* Spazio stati: `valid_cells (non wall) ×4 dir ×2 has_key ×2 door_open` filtrato per `(x,y)==key` senza chiave e `(x,y)==door` senza porta aperta → ~480 nodi su 8x8, ~192 su 6x6, ~2912 su 16x16.
* Basato su `state_export2.py:268 _get_next_state` e `view_wrapper.py:151 current_view`.
