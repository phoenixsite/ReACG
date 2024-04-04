from attacker.attacker_l2 import (ReACGL2, ACGL2, APGDL2)
from attacker.attacker_linf import (ReACGLinf, ACGLinf,
                                    APGDLinf)

# from attacker.auto_projected_gradient_attack import APGDWOModule
linf_attacker = dict(
    ACG=ACGLinf,
    ReACG=ReACGLinf,
    APGD=APGDLinf,
    # APGDWOModule=APGDWOModule,
)
l2_attacker = dict(
    ACG=ACGL2,
    ReACG=ReACGL2,
    APGD=APGDL2,
)
all_attacker = dict(Linf=linf_attacker, L2=l2_attacker)
