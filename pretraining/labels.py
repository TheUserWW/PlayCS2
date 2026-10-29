"""Masked labels from audited usercmds; no inferred keys from player positions."""

import math
import statistics

from cs2.actions import CS2Action, BRANCHES
from cs2.state import encode_observation

# demoparser's documented examples/buttons_bitfield/main.py. Unknown drop/slot intent stays masked.
BITS = {'jump':1<<1, 'crouch':1<<2, 'walk':(1<<16)|(1<<17), 'reload':1<<13,
        'use':1<<5,'attack':1,'attack2':1<<11}
WEAPONS = {'AK-47':('weapon_ak47','Rifle'), 'M4A4':('weapon_m4a1','Rifle'),
           'M4A1-S':('weapon_m4a1_silencer','Rifle'), 'AWP':('weapon_awp','SniperRifle'),
           'Glock-18':('weapon_glock','Pistol'), 'USP-S':('weapon_usp_silencer','Pistol'),
           'P2000':('weapon_hkp2000','Pistol'), 'Dual Berettas':('weapon_elite','Pistol'),
           'Desert Eagle':('weapon_deagle','Pistol'), 'P250':('weapon_p250','Pistol'),
           'Smoke Grenade':('weapon_smokegrenade','Grenade'), 'Flashbang':('weapon_flashbang','Grenade'),
           'HE Grenade':('weapon_hegrenade','Grenade'), 'Molotov':('weapon_molotov','Grenade'),
           'C4':('weapon_c4','C4')}


def command_action(before, interval, quality):
    mask = [False]*len(BRANCHES)
    if not interval or not quality['commands_usable'] or before.get('usercmd_buttonstate_1') is None or any(r.get('usercmd_buttonstate_1') is None for r in interval):
        return CS2Action(available=tuple(mask))
    first=interval[0]
    def direction(field):
        values=[float(r[field]) for r in interval if isinstance(r.get(field),(int,float)) and math.isfinite(r[field])]
        value=statistics.median(values) if values else 0
        return (0 if abs(value)<.1 else 1 if value>0 else 2),len(values)==len(interval)
    (forward,mask[0]),(strafe,mask[1])=direction('usercmd_forward_move'),direction('usercmd_left_move')
    bits=int(first['usercmd_buttonstate_1'])
    buttons=[]
    previous=int(before['usercmd_buttonstate_1']) if before.get('usercmd_buttonstate_1') is not None else None
    for i,name in enumerate(BRANCHES[2:10]):
        if name=='drop':
            buttons.append(False)
            continue
        valid=name!='walk' or quality.get('walk_usable',False)
        if name=='reload':
            valid=valid and previous is not None
            states=[previous]+[int(r['usercmd_buttonstate_1']) for r in interval] if valid else []
            pressed=any(not a & BITS[name] and b & BITS[name] for a,b in zip(states,states[1:]))
        else:
            pressed=bool(bits & BITS[name])
        mask[2+i]=valid
        buttons.append(bool(pressed))
    angles=[]
    for index,field,limit in ((11,'usercmd_viewangle_y',180),(12,'usercmd_viewangle_x',89)):
        sequence=[before]+interval
        valid=quality.get('angles_usable',False) and all(isinstance(r.get(field),(float,int)) and math.isfinite(r[field]) for r in sequence)
        value=0.
        if valid:
            differences=[((b[field]-a[field]+180)%360-180) if index==11 else b[field]-a[field] for a,b in zip(sequence,sequence[1:])]
            value=sum(differences)
            valid=abs(value)<=limit and all(abs(d)<90 for d in differences)
        mask[index]=valid
        angles.append(value if valid else 0.)
    return CS2Action(forward=forward,strafe=strafe,buttons=tuple(buttons),delta_yaw=angles[0],delta_pitch=angles[1],available=tuple(mask))


def local_state(row):
    name=row.get('active_weapon_name')
    canonical,kind=WEAPONS.get(name,(str(name) if name else None,None))
    if name and ('Knife' in name or 'Bayonet' in name or 'Karambit' in name):
        canonical,kind='weapon_knife','Knife'
    state={target:row[source] for target,source in [('health','health'),('armor','armor_value'),('helmet','has_helmet'),('defusekit','has_defuser'),('money','balance')] if row.get(source) is not None}
    weapon={'name':canonical,'type':kind,'state':'active'}
    for target,source in [('ammo_clip','active_weapon_ammo'),('ammo_reserve','total_ammo_left')]:
        if row.get(source) is not None:
            weapon[target]=row[source]
    data={'player':{'state':state,'team':{2:'T',3:'CT'}.get(row.get('team_num')),
                    'activity':'playing','weapons':{'active':weapon}},
          'map':{'phase':'live','round':row.get('total_rounds_played')},
          'round':{'phase':'freezetime' if row.get('is_freeze_period') else 'live'}}
    if row.get('is_bomb_planted'):
        data['round']['bomb']='planted'
    return encode_observation(data)
