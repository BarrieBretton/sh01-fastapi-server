# SH01 integrated editor — paste this entire file into ONE Kaggle Python cell.
# Enable GPU + Internet in Kaggle. Models download on the first run; no paid API key.
SOURCE_PATH = '/kaggle/input/datasets/vivanjaiswal/video1/military_seth_raw_cut1.mp4'
BRIEF = ('Create a polished, informative vertical short. Select a coherent story and a strong opening; '
         'preserve the speaker’s meaning and voice. Add brief grounded narration in separate intro, bridge '
         'and outro sections. Use relevant supplementary images or video, readable motion graphics, '
         'purposeful camera moves and transitions, restrained SFX and a quiet music bed.')
TARGET_SECONDS = 35
NARRATION = 'intro_bridge_outro'   # 'intro_outro', 'intro', 'auto', or 'off'
REQUIRED_FEATURES = ['voiceover','sfx','music','camera_motion','transitions','graphics','supplementary_media','captions']
REFINEMENT_ROUNDS = 2
OUTPUT_SIZE = (1080, 1920)       # Use (720,1280) for a faster draft.
LOCAL_ASSET_LIBRARY = ''        # Optional annotated media directory; uses Commons otherwise.
MUSIC_PATH = ''                 # Optional local track; empty => original procedural ambient bed.
PRIVATE_ENV_FILE = '/kaggle/working/sh01-fastapi-server/.env'  # Read if present; never overwrite.
TEXT_MODEL = 'Qwen/Qwen2.5-3B-Instruct'
VISION_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
VOICE_NAME = 'en_US-ljspeech-high'  # English public voice; change for another language.
ASR_DEVICE = 'cpu'             # 'cuda' is optional; director and vision use the GPU either way.
KEEP_LOCAL_API = False         # True leaves the loopback API running after this job.
TIMEOUT_SECONDS = 7200

# -------------------- automated setup and execution --------------------
import base64, hashlib, io, json, os, shutil, subprocess, sys, tarfile, time, urllib.request, venv, zipfile
from pathlib import Path

# Embedded, inspectable Python source from the matching repository patch.
BUNDLE = (
'P)h>@6aWAK2mk;8VqLIRB-*_W0049{000dD003cWUt@1=aBp*EE^v9pTWxdOxDo#D8UF{C9Z$#`QC6JXwbfLeDz<XbMSiL5T<_|7I4}uHI8!7K'
'BxS3v|9f`<kOW^O+c#~h4>3tB7K_EcJiBB})BH%%Y<w{_c${Z6HAv(ee$L5lNS$BeTZ2UzU2u}II5L8m8{?_r#j7Mv89wTprn$ct@OWjoZjfg='
'r><)-80Q(VisDSnO!xQp_ll?KCh}NZ-Y-eI4B2gUzan0F6UK`Lix%a5oR*iImX{e@RX2Igd?ivsGGtxIa{;$%B-<pwwip;k8x&BC1fUL*uyW{P'
'I!DFup7DtiU;Mqgx|lq8G{K$KXZ1Nv6WA{3QO*CzLONFud&Uo!=V!%Txgue%)F^dn1iPih+++E4!g&nK@;q`i&9^5rUQC?$EV~Fv2TkfN>57z_'
'o2n(|^w*rGSx+q~7C;6)!7ApAgcothyp7_xqCO)on1`FfXeqF`sgZy_;+%VxKGKBTN3I;qNP5>Z)+A&;$*8+zi)A$d)sykfaDUH$pT^Q73GrCA'
'DL9@^PNrjbJ~=y?%-z}f@#p<LDo)3k3Id_Np<(>qNH|%n$k2#lFyS@jhGXO@4f4>qU((1>y2)bLDcVMgDmO?NM^p;YTFDp{*4&@P-koKm&}g#2'
'dsNtNe?haEcxIW}Nl14yVT!oI38k%B38w*j7@3D+%Vy8W$RZtGo8WvuGM~o!Ihi+N+40mpK0m!Uznr)~oPTgXPtQ((rM4pY>B*P*_`_`CUQJIY'
'=U=Ye%gOQi*~ul)t;mD*Z4aH@ic^%BMr+H|cy5^rqM7dorfq`^xa@p#H95PQo}V>%zx!@5*u}ncE;4`nhoe6Z)B^i^XmaV$@R^rgrx`d!bp<|r'
'a|5eAJ}ELcgd^!VA@`PecMDK@<O>S~ekska3$iPCgnZzcJkR|-p9Y2-#rKwNIDY{n=gKHKg?JZ<=+lR%0bW?<!OzaY$~o|_4n7SJPKO7V|G@J@'
'n#D>`*g9Kj(|2^Ug}JUTX@AGF>4giX;(ng|j2bd+T%0Aq0K&6Iel?j*zMqdz-OH=<`S|-u_at2y4q<>n1V?eeEps|VfsB6}XW#=uKD<x@(uah4'
'8TCO|`Yz-k3&6{=l15Dt`ywo*^%!bF?9B2M4ArNjZ_!YEnh_R`I(Ce1iJZMS3fKbDjc?YM6mp{`@CI{9AMKyOu6(wD7$-N@D@C(kA=~@GTTM2W'
'rr_R(ED{mS_{jkg9n`O`QjC$zx3~<ja4A&UIBQT`LyNcp@Li|!UCk!Mt|EGG0coE`D4J#Wj6j<(e5_V{GBqNVZTsnx9DVzp-CxoNr5F2p`0nVY'
'A^MuZGH@|$c$(Ni{ZZX+A3?UEIt+vjaNAS2EHZ#&pJjBF!cNt>;a))3JWKCcwzN#ww4bCf!ZQsaQbKvkK-Lb4AVJC<4_j$X88E20wBuU{@=D+;'
'$>1*`n=mG5)Q@JRLUSlY2G6qer=ZW`7AVwJe+8-zT|ZwX7G93bUs8zD_7m<@(HJxYLpv)o3;xJGVI?~1<Bbc7vnm2Xq=vKD&6XShNn;#;A`ADE'
'Ji`<0b&YdW;I%1j6Y@s~QAax2&PpmqdWtD+H;?WhV#2mNW+@$ajMQwdYihX|#jM`q-n_?n3mJ8^C3|VqwiX4X1z+);7LiWCi83a@Dd02BYGlrb'
'L*o&I1k7v-B+~**8_IM6DFPz&aU-JxNIeHBgyV8o>O^b|PhMN3k0g}1vE~%0OIIypNe)$zsXK}Cgig!n@jYhCZYyltowPELT3DGO*R^}ksobuq'
'v`o5kl7MP$?LMC}-c@0_L-k=C-{pyr%r{^eFj0XgVU=3yKn=k9@JF7;9gE;BjNgsAykXcnoywg7#O7fJ+ALYgCt9jvkUL-xkw+~#Cb5j3jX+Xx'
'V&H-UimFf_*g+TNV_`u<bs!L+lv@rfVG2iuLFq=?nDVP9$y_qNHkBq#1V)u#G)CP#K^*V9Bol-sPwlx~OA?YnJIVv@rUJuRP0-=xSsVnwTmUKw'
'rvS!x!bdqn^2MBDRIpIf9VjEj)Noyo98!HEbrqVF)v)AJWUu-pIM;SGgei|Enq&yL46&Dt0OnI8vP0b+fGcNJKy?Eav2+QVqe<*7Wn|ShZX17A'
'3aMD#RF{?PD1+GNu)e0;N)jhlu}6_0H4Kn(0LvD_<e{M><mS1wKy{z^zL=*}gqQft^<fg!aAAlutoA&gr5@qF-LeY$Kw-1B2)85WHwApo4DfJA'
'a_CyFpftG)5M1j|Ze9fjaC4PhxIyV9);He>lmw{(0(Ns3O47cfm6Vzh%IK}bq*6t8v0^>yYkFywx5Ji86bnn^9+}{%nDD@=;mHDaClFlugiEg<'
'$>OQuEW2fGQ#Se^1*8^!*;Cr9XP4&_HZAjso;PYK+AMQh=MfZ|P)=PcqP!OYb%a0(Q#xSm!C(NUQo44jo}SV0_mS@EFF#}Hxhryq6;GOw%v+*a'
'ztC@b2FW0_B`^fldd~<c!Pe4NM4@zQn&vV1G-&O1qxv+(;*^U>lr5`-3@bq$aLXb6EoBE0c@&jHwbfQO(}rrkSa@6Y(5M2pHmS^omF3n)7ZeAD'
'e12^#;qd}eFwPzH2l|N87YwK?eb57}q*C8d^$_%7nE;XL^-Wt&H|HS?1?FQ_)xWv;@?kbT7Ul4V@#W-A&v^63?sHkqzd3wkKgp)iVO0!bA2&UG'
'l<Ih@ZL932xy3=lg2kmGY?Wc0F2M3AZ`oI8mH3JvT!!DOHbEc(*HVPP%2pX<EHHdi+acnLN<OJ{w`zS^JIwvTB?)sywOaK$U{TrawSXT1+%>|d'
';2R+$;OttdUjVjR@1*3Z+;&Tgvk<GFM<H7=w3Bj7%`BW<LqvtMl$Jg+tZw_qpo87;KK}C_f=Jo)kvtSQ7<F_h?0SCMeq{6kOaa@aDtQXz(h8UW'
'j7mZTI>k*FR4SZL^HHP1Pfv}cq=JUtSg+LvM}vXBdSk)SNMkLF@}t2~54%g?Z4jBudbD6rMB*Zp6%|ug6;`4Bf?9jIvP+V7ZLw>AmAH!amQ>$W'
';BD(8$m&iUg-avxw^3TRf-C->?;1RI25woYyc|7kE8KUNUFtBlm0sd>T1C=U{0>+6ZjAFh$q>%J-Y~0!s}*-Io2^;tV}0cvSyK#eJAek+*1EA*'
'Cdg*HrOcqOm{O(Ba*}$4(*|8je3jHF7Tc1a#aK}~i_B1sJ-rx%f+=>t3<2ytp1B{#v)PC7@n=+C1~s>7_rv%eQj#s26~ZjvmV@wjL({euzM^#!'
'0ECO;h(U2CA)LZ#c|_JQj*(?(Ji;EHS`8HZ$qMR!Y>OSxtY&!I#<#+^h7Gocb_2L)jpVCZs|?qHs;yx`Y}M$d*;)2=rgi+)is1r=l?+}~y{Y~|'
'^QEuVDCVoF<y9vp`mF)7-d48jU{Zvf?Mkj8OKf*)?{HgX)z<)%TOr|dX9~CA+0WrLLDikhDm4nCqO(&28PflD;0zsEfDDc5`uV8&x`_6-mqswy'
'xfHz~pf#)@hwHea-L%!zB_~`L#5XPD?dsOn0%Cp1+H0pWBI-0+W48#1TZXHMrxVl@ZzHl*8Bd>5?-&0jRNO|pYCu&w&%Ts>Uv>Y#$k$F-{m_R5'
'T*x=H#tsNzKIi>d5Z|(pWlVmZQ8~Zn_&SmGHJrYqH*z2Iy8sXYX1#Pdc&lyVuWxHbzVKEk);aYuae(;tbIXSqpFX;$=O>fd&MymetTH=wzsz@&'
'&t~J(v3oh0|2UbqiS6y8V%~JMH~l*$|1V2I(iDQd$~>_0qmnITx18`zkzL^&*d3Ff>6DKWNN-`<A@0TGr^)QH{vifBTEZU4E!lL<nEo{WF#fqc'
'C6Cf1B<Zqa&X=>xi`n?{lRG{>o?Kqm=LMW%K8?Dx`Efp(T%Lb9J83MZ*IQ-)fCx@QM1=+bSy7VWK{*lO@{>U)g@eQo0NaR74meW74#|BMJD9L}'
'9JYv+Yxs(6&C;BNPAN%~9Q?+?eWE|CU~vcBhcmh+ZO$am;uVPMIaGY<=B(eg*|CTs1$ce;Ue>}|9ei(*I^L3ccOihH)WP0u2$S100FQbU!(8a3'
'!9#0!m8Y<tpL=(W9;FzKg%~hZSY)9MPBAsNh*=tkh+rL`lK^D8g(L!}5=!;A$ip~V6ldZs^Qcn*2tB^ooWETsDP55WwI&?NQR%Bq2Vpv79?O*e'
'8SEqxpBi>(8L%b<QD2ZcVEt|JK?gpK(4Kh40^((D&Ep6f4<{??AbISBWCOfAtS*884(qLAVX6yGk|p!ngd}1sp<@qta^isf3HEKuhX8Vd1E@HX'
'Z!xwlTG+Pqh^~F0Oly77B#*qM!#Y?J9}j6wJ58-=OfmB)ejM3hlVQf3HGw0FGDkv#7EwmMrKljCU%@6>w%KNu-~ekHA+Q1pkFuoWq~ug3SinLc'
'#iXqh*1G2M7=%_KHknY%s(y_x{-z-uwwou_wU1w@ly@Nwvg(er#Zuggm$ITAN!Ju2NNvdljaLv`Fml6sd<A>{7hsNu!_DD6^RwmQk|G~KAmB1A'
'B3a!gO!;9J=H%|A_T%(qat?PVlVcPCqgAbiB2WP4%dH&#em?)~o=oRBz5Hx9Etxz$1Bc{Cv0H(V9R7bnpsHcm?CYz}A+`$(8Ea}2!DHC`-1)g#'
'Wr%WBh#EG(kA4XUS1n$ArM+X$bEkt&QPrx4k9Z$JIM`jbB2~R^E#`g2GPd~UhW3Lz3|IIX$>OF8FYfig`P0qg(bGTstK_{|3HZ9NHMK8Er|K(B'
'{WY)RS$>of$Jji!@V^Oc!I;~e$u<s)Iu`#6Kpd`VQI+HWK@hKKko#e^ktUte2a@+$3aW#w4KdrzVlmY&K4}#M;%!7~)gQtvak8(|(fiI%r;w50'
'90;gbcR~ICG+4NFPrdTZHVGY>BBvhK&uvuhdw=ggP)h>@6aWAK2mk;8VqHcf(pMD)003DF000#L003cWUt@1=aBp*EUt@1=ZEtRKE^v8`R%>tD'
'HWdAC!2f~E22ffQ*$J{C(3%vfTVTbCZdg_fLy#J@L|bemQ6s6iS(5+0b1BJ|lMI<rsNv<g51#uV_xARFk+%m+BPIJ&U8_=NO0bzOOI=xJq%%t1'
'h=Q$^%UNyojjTkKN+zm|$**;xQso%Fva-^~<I#9@A%#hER!S!_;RI8br9dkzB$GEPlhBV?r5)MRONM_-k-7$6j4a>{`!V|?m#omKD8{23W!udg'
'CD+n0>r~M%Epq^-x~#R88Cx1rA`bRZ-T)|^69|P6wvFoVMw;94$mu~$nY5`<wNrFr?5Jv4kbDyiv#B=zbbEVy<I&RSlFjGK#x+LH=S;!3abN^T'
'65a~Y&`EPw7Mq2Uopxi2LM`H27%O{y$o!g=$D>RxSzQQK1z^9CVKN>u{E(2wRNiXbR0v$uGB7f>%_1=T)#c~2tNk+`G0yO@&UD&g7-OO8ARJgH'
'XY&mWHmJpIp+y$4YgJ`ihqfu4WvA@&s+BCYf$j=%SFBTsSJDOCa|^e34`C}Hl<fpQ4jc4l=?9)Q>;?pD6;z2oNz6kAFPC=@SK;OKYO6v;1r>s{'
'8uF{_3$1mu#bO`OLX?Y4unJ|7bcB+E*Cyd0TpuEF$HRe{uD0g{hCPq`(8?}y<n1Hme1QL!{aO{bsOVA^N*EB>Ajf@&L~SE&I%m*PNG+n1Sv+89'
'>DvVPErm;SWcYJ1n@#U$Gxp}M;B|6vaUQ<L|Lp#|@cy@E0h=t-CCr34$1w7UjJzMRPrp=GPF0N@5Ul(uw*-@*<zy!;DiL;i$~g72=6)P_U5Xcl'
'ao284)^bsjdb`rA)pk+V$Nb^NfQG(&c|7OKZ%n?_(wlErbv0NZjQamM+n(I2f9T}Xke#uU>Gy}zqhtS}$}CYvTjW}0F89!Apr1>%%AFU7>=ipY'
'p1$ZGBVdPompA1CLbSs(JjjBE#|G?ik8RNx@$RCYrn8?t0sg?KNOK8#zk#dVx!Q6vu?xwyb-p7A)YN_mto!RD)@H?DJZv{vVo}$!%7QyCtd(xA'
'GM@0Xy?^F>pJo^hh)X|Buz3ZNxz&wHC4d{BJlgfgopJ>bZOd;*Oe=1i9N)q0=ZOcrCvx8o#NlJhnO;|P;5l)b;Ij#+m2Al#Sohfhuxm=5>~!ZN'
'5B;7uH=H=1IxD!?j9N5<u+NQC*70zk|HSyd7tIWaaZlr?I^qd^VabV=?M%#Y3#>Hmd&|RqkaqV?m#gjN(O##4ANlQtLGz|m?)0P|GMhUOAWOPk'
'(rYA^W=y})bqf_*GMRLl-z}!edLn8S*Lgimb1p?DjXk~NpR6<o=POw`l*Gl))58z+aCnE)Ycn~Dr!45ruP}X^MGP;KH^ShoZ*W4VtBK90hX+ex'
'9UL9N9G(v$e;9Z}@r#n2;i=+j@9#T~3lZZPf-e^Gh$9YZ1YASZ^7ObkeHvEQbi0YSw9na671G_-GU05W1^sM&X59ML=L_XZQ9ns=Z=EX3)`I%v'
'$A|<zK|G=1bu^9qDvUU;pc;{G@80LkDb>MHlJ_)x?`b^V?_9qCVYeNI$>Gt-^dWLU>c*W0hd)He(=gmEg4hU!DhGda@a`HLbb#(z;fth$61V*}'
'2;tHg5IlH;4CakD%YJVva{rEiDDLh>+ftseezJJo%g)?YTZ^gAFd}$UT~}yz*6xMjZcF^<^U1%S#-o1$P)h>@6aWAK2mk;8VqGnQV?7Qe003lt'
'000yK003cWUt@1=aBp*EUu9%zbZ>GlaCyx=X_MPFmfusA|A1+46(u*DeZ)>&a_C*haVFkLVz1lD&Xz{Sq9{mWni4fU`k3DS@B0n_1bC?1$A?u_'
'Vu=JE9v<F}2lT<g!4G-8UC4!q>nh*GRhkI7NUOY<2zi+<<TjDw=!MwEMFB4_>BmBr>1vDL;<A)giQfunEw6&Fzxw*C-%DAFYAuDlPRlCYuEYjF'
'#zJlvyFA@i$N#KlEq(Fw{Jpr!iwjwZc)JiuzS-p^4jqWsm$JAKiy~fDUw^fZOOe7T`PEh|(n2QC<mS1E#WGGaz#NAz(|jwEcq_INx{4hBcU7d7'
'Bs{Lt?H18U>Ix7%I5_zFt7VaIL=-LSsxD*{i4?{zD!^l#S1`|fTjEUg)ADAUq`CP$k4yRO6Z7-WGT)jXd1<~D(tIt~b(Lmn*j-$$vvjV<d<Q>q'
'FnSlK5z|3NWwMr=ShsqKlzuOPbY<m>9{_`Mx!TAr{CK4Z2Ux$$;;k?KTEx3`n(&YF<uy=qTXhH6<yo5C2!m~zXO}YK)lcJ0z3=u*3*85wz!FfO'
'Xa~b}8~?M8v$Qg!q}y@_3)uAV&7%VNLl#{Kn;d7R0c_%31WdD__4)d%g<Og<zLX;*sxL0%td`@+*Ix<vz(T*1MF~9EuyPDco+fdY-H0-Wp(|Mw'
'^^OL}B*0<D&Ek?Lmq}ouB}_3Ou0g2PW+y^g!{92d)=^n6m+AG$3*dz}Hf;i8zo?Se)d(3RSk#+cIU)pc&VWu8p7|gYC30X~CTSY}6lbLz<HICh'
'0LO-2T`iBE*<cF^R7-fi<Aw%Fyo={4P&$?6s9-kHbSNY+`BvQWFZ_5}4m5~ZdcBEI9Rhj~m3dtxvJAKpaB2~QFrtXg^E}hg0}F;}l!NG%!3wCh'
'y&QQj-$bw8{`&6i`RnNA`T6S)(Jya)c>nVK|9IoEZy~dj*D@PMgI_n}TLZj%{rA_uoJTL;y@~$zx&t=7j_2{sFkF7$5BSaBU;gm&e+D4ez_Gw0'
'>p{@EK`-pze>;Eo%gghhqnEE<y*@whfq#`=q@*$j2QV$12^V=H7j*%vQZBEpzAVy#``+$Z5f>1mtf_)4&>!m&&~PUUZ%p$r3xAoXi9`*{--Z|c'
'!-w<e*S9~u{srdeOmI_|X`-1%kF3G-%eu&h0ss2@`I}eKyO$q+Zcz>l0qC2Ejgbc2595zF=WpKr7XAE_MiGbG-}d+SzbIN5Qt)`2d?7P~$(^N|'
'K*~|?lq1qvmS*|@>sMCsW(Nig^q)-DVlLyVd@gqNHmPbdLdUCkSHfVMINcf@<|`YQWO2GF8^r_)yw1QJBy|C*v;rkHo%!$|zg@!Y5C>2_p!teQ'
'E*K?WT*#X+i#PK{EUqTvYU%-6Mdi&jviM1t1}w1xb~O#fk$yQnI-N~D(Bts%g$TZViZspScBI~q#fwm!J+UD-m<RE02VG6u=gv%;9_p!Xj;M?s'
'ObZ#5GA@9B3vU|y<=<!j=J}$m3TleWJM!?G=X=0C0lY|eqw%NX)X>HP(cj(C$K0ghm1Vk3%QYEAEepP26Um}3+C20M+?1+7-XI1f#|;*=+{Qcf'
'b-<S8nzwM(z>Z~NRVaT?IdKV$)9_H+_7)PblIUh=`x|ZZVgn`}{RcAmecj}C1>H;+;4o#f6}{r^&8Q46zzT}Lgu-L3<`IvU$e;*0KrbsQ*~uyG'
'<$)QOQv89I^mS3>#mM_nd4&RW2RK_YgI|K7f|#T`kSgWCf}Jn397q!vQtZH(!oQ%RZ^-Pk2Z9REU}}tAwu$h2ip`Oj;3ukS@HjB%g?M4wlV||;'
'Nxof>?Q*%x^rCj+d$CE)q6YxU^u#&g;!izP+=#z7^E*$*Ux0K>HHkArH!59d0+N#iPoBf%mY|!#kuJd_0+v{SrYttdwiq@595f0yntF%{h)IiK'
'T-FYVae;4kH=~7=yYNF%qXXnHd}lF~J!_T}nJ8Z_J!_@}-B_xU<W-1AIJ=eHO6b@qUVwnhKhGDy2SuD*FqlW;H&V2?vYT{q6(g5`_^eQzbo8<a'
'#E&^~!$uY>d0a?%2AvR~37UdN2dh=-Un&d#q{vI6bqMMne8|xit8nZ{WI=*7${JWm0kQe<du&1K7-;%izLl<`G-FWsrG7LZU$j9ovmT1ztfee_'
'h7P*`xsT>rXi?T1M}Cf4VzG~@8Q2CDEDmPE0&Ud>yO5Q|!HsEVAew7AC<+JTp@xC`nUq`UiNy6H%|#fZcP2}aBWW|~$1sFVXM!57Q1gf4G&pGw'
'&LLB@;yAGElaEK@C0K(cL_%|5c!3fq(o+1h&QUN(p&O0BVxs*HjJwf6$$&%@<OB^=R0f-RL*#R06<!#0n&xJNke6jTnQY?g(Wy^P13elaAFD?S'
'_INy-PSp3AOG=EYu6+W9<doZbLxMii<Ft)DPz(nmx@+vnM^W%w&53;q8DF@MF~)-#c)|<*%t4Qnt1)FCV`dzTMtl*!>E!fmHg@5}0OTY<q-Y|@'
'H75c>ABaG+Jg$fy)00`_pt;@Wg76+`0rLUip+MrD1y7lxi0SzF^x1gap#x2V!j+>KeYduVlWf>@(!PzAY#5<M550AsUwD1f@QAyjBu3|}3!XRv'
'rxwN-0x!b3PiTesKa%K#T#?YL@?HM`S$-vpD3i--Xn+_NmdF!&B^WswE%K{vKd>E)f{x=&CPOby5*T_2DlkQfAX$7O&2JG)4nWb)J9<=QI9+0}'
'29(@PZsSRCw!AyKozuUts$*M7e=PDH`cWX(X$5Pzia|NiT3xLru=urlV?>>DZa%s;nDjyAzR2r&22P%%`q6b+u?L3AiI)sHbY&Eg5r`Hi&`azP'
'DaJ>`;wTm{yPVFGgaEEL$M2l%0KNozEn>h*Wa$zbZEPKqCrQ0aNhwTQ0uIiq6~8!GI&sWuB3A5z^+-Gb;ddzLT3C9x9UL$~zn1PB?9KMgl{J(z'
'_0G43I8@<}3|RKgPI_byy$3UEo@Rrgeu1$SLhr#;H)+BR+t52?5*EP8*^^nvY}LD6Mp6cmdBovuV?T`NOE5Vlp%i`DGHY51vtB=)OwL$sH(`E!'
't&>Ob4R~n2_+~t7O%7TXH@0dup~T2MGMd&^u;B1KZIc=m?0_nL-lskfzX_gpR|^-E5Ccmg&TO6eWJqVal+w#&r*s)&NdNcSTCiW|u(CzHl6#JJ'
'+@%R+wZY`7ukijl3{IZ;H(~JPnSYfos&yDVd*ZKUx>{FZaQ4kZ)n$!3aSk5zGKHbuWLb?NXeHa;0<;1FTC&y)!2tEcf4E-sDY%1ZqKYWa2&Y~W'
'Q|7j8l%~bV%j*HQg*Oo?$I@|!-fpuSfsWH9g$B&}umCVS^b!Ms(KZG`v<s5NoKL=5Lnx%<2mpU$>seH6<Z80Mx{ZUzuZ@kKb(!N%{I9rOSq-}i'
'8E2_lyg0eD7?jf1p*VUPxUxusNXT@xVRb~kK+%j&gQqP5qG^a%EAU)Pn6$@sv6H?rY~D~-v$)=TP@cgRn2i!xBL<u9U*;M3XQ8bV06@Tx{ow2j'
'L|EHZm1ond;k6~(ntkOYSz3D|D&C~{79B$?2F3jRXl$j_8^@p1iS49}R$D6ISb|CcQ7c1=vzjE$&tUz8N*$0-&z4XF7guqCn^Ty#!E6MI5_#Wx'
'4}gR5w7O}sG-+7^BA^{GgfCl(zAqYIDRswF9YZ|X<JPOS;Wid)Fm{>~mfm}vKDoUQ$eorx{`p+TV$jvX=B_7A(+<`K>>2X4T9QUN+%p0196*5}'
'(~>ASCL=VjE6gkupaU;sOp>BZ!_t-p^@kuk6t)qHJ%LK^a_HB?6M3_xJ*y<eu+w;Ax}QnkD6v8Qx-A3!v&A$f2ZI?aYJSJv!82tS*n#7vAi=pe'
'(O>R-{u1T2^W69N!JVDl=a%y#U94oc13zzi-chD^C}rfs`iv~A(Sz<$LxN#i{+e<I#Y!(KI!uIj2u4uHglq!3!yK7!V+$S6WUf3Ub!!%y?INvn'
'3=fY676rY^Rz_?*JSiio&8%<_9vHk#AeOBxUTy|$A_BB-CcFs4{lSH^%RqX~pz&MY#5YPIAgJeh$5`ahj?$6rag3WRn+g=1q1^2Z$1Mlv?ey76'
'%dEACqT*b!l@g+va*YdGaNYz*G;+jAWIziUK#wq;JUf|<_aJag0Yq@>>j2}(1Oen`wq45s<Q8d(kKzJz|2g>2X>tK>O_dXVLl|Ku5sjEN#wR4!'
'2rcc-ZR^dP)2<~+VXy>Vh_h@|c>h?CT<r1RIiqIV_y?@1n#jO#hSFn@>o%4>{>%V<N!cuF@CSvSVLZo7qRN=DAE@EG^gcatSb%eX3D-6ZYSRZ#'
'P<8m6N&aS}{o|);58s(8V6FH`>#5!?I&Q;zh|Cg0n{zEi@l(15-C*sa0;n6Ewm;H|-sh@DiBj{s_y#TOSZDfQtz~i{OkPvzKdoPVLORK7bl=kL'
'5(uZ0qy!AsZi|r*RqO;tsEm3}O9FtNp&cgk>u8zEY{BNSQTvnTd#OreByL&K&OZL%en?y9@R+yGn-?zCAYQ_Js4A#|=IS?Xc{qchm#wT4G_!sU'
'!#3{2#C&zocKk3^+~7H0pWWO<b`$fpd41;!ks{+m>||G(a9Q+Qav0o2nW~>N|D-SK8M2!iC0l41eUugTwxtWUuo%AaLvS|J(I9~=H+bxi`Y7IY'
'7smr3xb+#e7|Q@@B0Czdmo-ge%g5+sbS$YLa?`9&(3RCQzn9EQ?-qOAO&Vg|>Th?x-o%B6eFK@GK_R0##&DV3q7`RNMYHLB!#i+xC5U1<)jFcA'
'aW?_spwz?KZ6@nsFc=Jw&|MXa=~?MgFAqGf2@c2rz})f|Z(<(kxYL`caIz)PWs*a5MEfx(zBoM!&YnKTC1wEQqu}IQ`pvz8Ekv^+J<;E+QW&n$'
'?8MT`0u~1F0ry=leeu{=+s>*Mu8E`H(u6T6#JV%_7AW%a%@JmqVI*|C{`lqBiPqp$IVyH3R*tdqu5`UkNj0hn^tGx-MfFTubHx#*tX8l>i$J_5'
'S^}+iMP5p5r0j=Ph(lIkp|jWIu!2{JSnlF<q2?@JzI%h+At<ajoZ&UeGE;SnC5eHVtcsyDg5f|NlvS66#Vp!fa_{w$>yqy4Z7QI;Ae4eMkf`>D'
'*h*uDii_-dPd(J6SJITAIF(waw4z;vL*)wZ!R*57se*++I1*56p*oggcwUHOFU`cEu<^AugOQ`1EI>KKs(BM)NFNgin3Dp_^b0YfT4N><Glb?L'
'ZK<@KAy)5HW4O};9$HA`sfB7U;>$SAD6~K`acNKAbJH~30po>dwwu9VLNVoUQ>p+OrH@BepQF?9@pohP!BaA&@pdIg&&GO(04t5^L?xC$2mdHB'
'7MxRoHB3U)im0(D!MRmQu0Z^NvON*EB;({Ap3Lq7;aRag3!C`Q^S8f=+XK=)2a}WF#6MvDaxj_B`~zL-eK0wwtok{8i1aDz`JW(GAn`GK&wdk@'
'nf=YOUaeS%mhn<jr8rnpXt=^22Q9oPE4<Kiy;{elE!2R4_)ws?3DK@iAj*!QU%1TkDqevbPzqS%qGrvVqdQs!wCAYEhIv(M!GIdX+?3lFC6>F-'
'b4(+~xaD6~tSmp9pa&R5r%MQHGy&<=flFv6CDG8HW1CRyKogZSASKlz-}qQ6Adi)?!kwqAE@^KC8AVr~D3-pAVK&u`si#S|msD>^6%_eg<=$cF'
'z+!u?njC(2i_w&Ts$w*~mJzdR(+zywD+uVrc8ifUYi+L7__%DglGBCsoXKxqs3k`cy!EP^okXV^D21Lj{xh+LxNB{B(tyZNN_Z#F+Afu5DCH#d'
'kFcbIQ}Brp%G5*Y@n*EM0xLOw7{GJ`%#4M(0JZr(x%KEDZxL7Vgq|P&xs$8sd`iOi4*5$k|Mm&xwB-m-Jd6}3SjUc45-6zA{H6kekH-P4z|r{b'
'&QZ$c4)<W1>JmyzRwHeVd^(B|-g>OhCw*24f+bm&fFw9Ruo^|4;nZs2E(|+(=H5hoxtrbZGj~2fanMQyIeOmY1O|++qbk3U7|wima&qS*_4B$4'
'N2gEy(`U|(u%2T;wjlxzz+8P*m!okHtD4q0@6mYbC2RD(OK(P-n`)-;%gjty1Me-^B*Uf)1Gxj8TYwRrS3$Ik3+PRwGg56o)Uuyg+UFa$GTn|&'
'eReK+OO{4pznI&Aw?6I!EgowxZ(=o9kNhcCxp`W3c@wO|v6`6|RQC&7^C@agc2Mbrk3ct-UGaz<$Pue&Q|YyT1Bkl#)_zXC!#yU;g?&%3MT`fP'
'7PJ9{f<1V&2_wjaP9zC+>cgC7lErY%uDh0MK1s6=hQ%Q{s5dn!-XUZ6qaUD2A}f%pY}Pz)wq6I4&pmO{-rCLhDo!g!zEO!LH1sU6H(T^M(q^YZ'
'm+^+Gr#i0{RC!dbQL&8LC=ae>YZ0wdR^0{<|EK{2er|qp1HFCnIdz1f9d<Q3GE~ytQ_7FVGUlp5_S8<zlmhM?QM&5z)XDP$G_1^QAKSWTYm*T+'
'h^nze^{=IvR>;y}_mI~JzgaZbn_`^UU`#_hr{P2nC;$jK$S->11Dk_K)UesnW!+>0>|-vIv-d3A_;l0&S|-ad%yWH`_aKGxQ6|S#K59v9EyScZ'
'JWB3V&qSu4l8p${ut<uT&FSi)Roftw^%ld{Jrr<H_W5+uanuaR2gsBIz=#H>`lhwehETTZGW->4{vwK-o%m*`)*GjGQl$*SkKOQp81$x<Iti%{'
'C(S6~YslzL8j?B{37EUlh<Q4}VWmKos?gq=9&hYQy?IYmZdE)mo@B9I#nUm4^;YSzhBklZ^R}tmZ;?iELNR-!g~3@<N;b3qoviEag<N1#6jRid'
'a>ySfAC8~m_gX=%m|4z!5!E|zL>3W~z%`7TND$D$vJoHm=T!dt?fbt)KfZa7NqL-51mpojg&Tb6j=-;}2mg!Gh2ate8f-4W{Etx9wxDy!RiaAH'
'D8Jw{5!|*B>vc+TVaqfmcwQ{*KMguie=v}VF)2#~Ic*)Rn{|>Hjt*`T;h9d6G&u`!fYf$~l_(iRu2UOgZ6R^PrL2;*TEu(M>H<R)z_x*j#k0J^'
'h7~R`-(z9ct|jO~qix_%FGPXvz565&+1VI7hmvT`E;yTWf!#;$iC<rzx%JH{S-&~e>RSg;$KBII14~_Y%VTh>9DLagns>ttTpD8?7;e}AH38;4'
'3asVbL;1#e<U9f9yTOC>J<O(hcSOjKE1(#+*c}hVKA6cZbf`Aq9Uht`fUy<;Iw$L78I7Fj9rAk+RPvo;1>7P{nu;3Q5VQ;Ea2R4N+IiT^ld8mz'
'h-c+;)OKQ<-|^|g0J}|X(9Oo4EyJBXmJaaN#!4UPW(V@Y1y+4vdUq;^6wW7#e3vTD#_S+OryW7U<N&OdbC}!Sylqjjc}KmI+BwniNPNIkYZ8kv'
'N?A}d;<!9Ros9=1(1*ZOR=(tm68J!ZPNHs_y%2`?UwG38Zmz&2(toj#Y`|0C-?G#Iz`2O)3{OAPzN?_aQdnP`mNqyTvXG$EB#ElI)S<$(ETV(E'
'D0pB`RSjh~s8QDGJmno#oReLjYj3v?(%iH$D|csPQP~bt;d{Xv*S|!a<H~9PIBd$%81>mDDTfQKn9{~w!{Z4W>_gR$sWQ+#*_N5oDL`4TVyh_4'
'3^m?se~q|w<M;>2)&p#^>C-aDEs&vAvC$XBsbj=Gm5|G_41<#=zU>NysN59*m3cS8i{q0T?-qSCebkWdu-TVlXuMd4qDR6aE-K}>j>S3WgZ<Av'
'{%cl`4t<6^<UNFcz1dYanhaMU+xgW#;%dK~w6IO%Q#IR2;}Z_y$D8KP!Fg)R>A7`W@@YwD6Z{dKv(?!!<v1F_2Q5TLk+BAe4=3ugz2c9zq?K?R'
'3ZjTDCfKOX#S(vw&(MnYgd!?r;U%StYWikQiO1Vj4dk-<rjMA)ZH%{Zc7us5i=1a?KH_ENL{v4L4z1K?-VoBQSV<vlp$eG!r)N9`t75XX&(PjK'
')LRDPP;nQ(uZkP1rL{@32ScY2z{-K>N&rnvRS}@d0jwZMUMF%_iP!WGjuqpvx8w_LHxEt6$z_Z7W-vu}fW&w@Ipegl!ovbGwIJBerIHtgqP^2a'
'NdgCgNn+1>x1e-<*c7eHO}xd^`AX2$W>~#7yUNd1;+Y=P&0R*)N#So~730=U0H(u2Md#FM`jk=s*t{pJN;~WMNW4>rq(!;I4HG~u&hk|aNU~>W'
'2MZrQoQoW{XOROyFKmxg1t3pVgQRnk&E;mG7t&I5hY52VYpFdNFq&AtPd*$TtyAP)E|FB1)*e*<6$V50bZ_1}UACNtspX-xxBdEb`Wtd_oRFkt'
'Qk#Nm4cNA-&J;T+NAr5IlFh9ap=-AGw{GMbw1@Tlaq#46@91tbi{7mlM-RuiZ<yvXBaWMdx2&_wa|$<_YA-7~<O;E#!~PVHstp^e&8#HEq<ej*'
'#>aFj#_S+gMY=B2agicS&o=fCoSV#=vtHft9|o`8-B7Ke*uB>w@NuZ2F&ugfFLa-``A8@i8MFfs3(Pn<25M!Fab2{w!h3SmuF}7_B&4~!kd;dC'
'F!~}2!=lt5RkV9`bogJ*eI-pVEcB;H8Bo`xY;e|Ub5CB)MCg<=>0NHOsqplDU8JW-5ZrsbW_&(yWL-o?ZB;(O4tLZEdrS48^ExncXkTr%57ll4'
'LU`t~7e!&ZED)WbV^6!0w0rLPx1I|2*)UNFIz-T%?f2L_@g~Yk>9N+{#QH%c%qH;gCP**eW8}Pv37MKS{GG9Zzs9>`AW;q0I?Is(`OY`I%#3et'
'0%Cygj?oe^DqWRn*+ur{6~pq6?tL6ovJR{gaRXOFnyVJ^VoH2K&L|xQsn$Uv(`+;{9_9F$LD&&2wtmrJx%fJ24SP?FyY$Z4YvH`wEdjn08i<-)'
'i3frb^hBiS8>6YW1Vy0SAlft4tY>!P?cl`RC(>-O@;xMkgp(GHeLOo(#5klbk9lMt%_H6iz^_L1?(lT{*dZ7(Wj)8FD2(2B%TN4$*j)7U{oE=L'
'(&tA)OB5`3goBRo)^nmwZvxD11)1JtZdcjt*?hm0hPY11^6W4;J$p>kJGP1Bx0WK*6lg}{oi#9SdxIKhH)o_~J?8Cq6>$><eWyIjQKnd!3(Ay6'
'vfejquhlIAb;~N7Hscqk_7zQ7Q**7#y7XZvZ!#?%!L|-ey2qww5I$6Ch4n|ffWF*O+Z`p3%$U}4vsS?9<=z#i;Hk-K(a%HgAKP;+2uY5)G?3Hg'
'W?y}trIc2Iijgs6Wi~QKl43kyU2Lj|fps-!RF3$Wi`>lBS!dS9AGOS_5J_E2))07lG8>N6`IfxMgh)2uKhcb=8hz@zh(jwS`H$_7d@*0#-BIBT'
'1|R8PQ_;G<vdvsHCjJH?h;aA}g1uyo3T)w4Y*NdCY8~t1G}F5|@1&2<FB%M}^rux2nq%2K@I``(9Zzohe7)2$rfKuKI!>@=Gy|XI7?TSb?h(^n'
'F@F0XwJ!JL0$wUQ6X!qu-5pSsjjBtj1S@pX;#5bfFRel@4a6(UI!{`hj7dUwNzUzQV)NF$0jkObc>mPqeK$M8T!Q=AA14V7clp9q#3AV_TRhwl'
'vEQ25mll#?aLkKypW(g>0%omZSbK_YQw$V3{W}7w`#w;pKPCrcU*5Ps<<s%Kl`!sr58b2*Uxa9G)5Mz33^|StLz-OB>9^(}<Xo2SWl2?DQQ3*c'
'vbk$B0{&&GEDujDcSJvJToQs|`yzbsC8VN)<E2MGqg&H+a&oq~^PY1vRdmo-XE3A=n_BdU*UQm|XNM5XKQ@CM6L!8eFQQgx8jL$kVBdji6sh0V'
'|8~E7(@hPSeh7Wj)K{`ppq@e5c|JeD+j(#s!JKq+uWqsH!*s3C25jY$(wp}QeXAU$HG5jHkML{oJ{xHwhB%{BNtHapQf|RVZ;7WQ?E>&Y1HO>9'
'bx~N)Jk50!&9xxx^>qWdc8SEe6~_-0Z_v7EoVEJclV{VPCC*nb6XtAb)PS#%!s*g)6d!fb$iR^834@0hH@Ef-Hbd>I22<%82ccOT*+QAOJurJt'
'LsA2Eb(`Qqd<wJe=b=y+L#aTUe+<kvQe{oVW5-_eTvdzg>voS0!Otcyq`BjZetZF;Fj)95KCQ9u6BGQ@rCdbm!r~WRf^_i^mPyb6Jq&<F2nsFS'
'rix=UHmz~Eo1O-P{0QL4VHT$C6;kV=!Q)o7wm2IDT(_dT65++v)8M_?o&hiow*glYT(KQEk{E$w+V*OV-eHr6?=2s~{D!W_wbYV6t^{Ja|E8>&'
'I26;&YLYbzIdhGVb;KFrmM!1MdaHb>cwT}Z6<0^q`1|0Sr_DuM3cIlrxY2p8o)XiRnpa}dwo#O{QsUW|V3W7Pt`J1F@3Qhe{RbY5z<K&+eE!!C'
'()N~|<3}aKx#tv57|;4D`|$v2QfYsy?-tv(P-=+kh_YQh+lt-r<2u!Y*t16IJI%pvz#hW0vGq-@r7CW92fg0`wdmI4;xU<eniIdcUa19%yvY9R'
'B<zo>4sDKbk>}-2jO*p=qGcjFpOsNA>(f^BZhx%osoF}_BQY<kUjw~6-t~L^mwOl)^#^_MqK+y~x90LQ3N>j_T<`kKy=KJz^jy<RgteJB<Xzn1'
'R+=F#d=brKu8L<@oKbFh{R4<SXjN?aq|2<3uf55%cTGNhJy?nJu!pC3{~vklAIFfi8%;HgpO%gKDZN$+C&d8jz5{jkp3+q6kMO7jgB_^zjK>J*'
'9e&U2@}2;_kA$C=oT7O@Br~Ny5Aj6s(|_x%=13*9>ckte@HFo^S%9{PHX;@Kv^>wgkf#Nsg;^QnGVD9;(yBY*5_uk-&LX;pJ?uADWs@>-3J3q$'
'_#vua<;EUbaMCD#H7Va=%P*nkKc?oy>fYAvol^IQnh*RuK-akVbqzr~jobcwo-x4But9K+417VmfS&=|fS&<9XIQM~LI_g=)Qta4QvtMhwtuwF'
'MOX=@dncqd0;5K%fWNxUR(VcRn`NUOMKQI`AI&V6uD@+_3`w_b-TpD`0ggK=sWqKPAY!UArE~ih{jC-b20+3u`R}#h9pYJhjgbRYgwwgd-nr91'
'RE-Zr&?h9+-FhV!#SC8D<T2V*7I!~IjQe`uTt7oow5P-!ox(EfdFwPa8u{S6Ud=^EECJ|{`5$(O6yaQXU8s1L(x$j$a1wlXf`2?kmg_uQbgT<v'
'NI5(nhB3zAZPY)JblE#|%5|J{scpzL(khnJD+YKp@(WVEX57A;zEt-1R-a#|fBfOdi{lA!+p_LvpdEznIn34*y%}#%hdZ7*0*O~U(Vu_nxR7nd'
'Z)v8_aY)sdyLOoRQ3j#qg?``bgf{!Emh*Sp!A$*;t!Q9P?);uVwb<3EpX1Jw^^&|d$BB5}C1~9n6*OgzdQ%FWM`-z6AH(c%d=(XFPoc*!_xncw'
'kDS|&@}$~w`h^1MVSnSxco>#v(VE0P#rSyZ6ERE)qEIgJ??5zvknFg*ES$?jld8D`w@1)g`<dz$->a!@mk#uG1vuTH8TMT<;43TU{zV;Dm|7Xk'
'eCL{hg~T#6T{x|_q7=C3R(+wZs9>{uV&#|BAD={NJ5$kv8#(wEX!q8XI-?Xeu!QR9H~m6fIJ;;69e}zts-9=K38g02|91gZ@H8Y*t+T||*uFUi'
'tcK;Iz`njr!M+s|7o|DZjpLfbgNelYY&1yt7nGM-eucM!u1`*nO{wz{&*wkRJZ-0V(dhdOHM@NYL#bb@Up<yd<$JX`(jAnUnpwt3zp|_o-bmvc'
'Ca}#`w`tGNE#FIZ(%g(ER=Lnf88G#le&Abk*CAFLjF3wLtY)~7H|5CO_0e-dW^WarwZ&NE_{V_<raRU3elx#i550-{(lcWF8ion60f19ZQ0q0_'
'deeo{IY8Pu3fO7U3&p+IU^2~h-M{<7zrC~EUcy*MK$cTr9lwth3oFk+{TetjQ`KeYX@ltbDr3<+Fk8^4fzB+1gahSvLj}Cg)zWLP+e{l)nQ5dN'
'6>`rk#%LxluvW;lMsfZ>P)h>@6aWAK2mk;8VqMrKq6xid003u@0RR*L003cWUt@1=aBp*EUuAA*X>MgMaCz;$`+M8Sktq6mzAyg+f}9+ZHb{}O'
'>^Pioc4b?W=*F+LJd??CbP$LHB}62_0H7YD<^O)`(Qh<J$}^L*d%w$MEE4EPb$4}Db#*<$FnpU&l6iNQ6=fA1p9YhBvC7M=%JOB9E-$lX+WYmd'
'e*LS@S6LY>^67e>2Gew&U8F@)rStosOjikhEraT+NK>f4oTSCF9Ad2}L4HxD#Z7|E4TITiRpb|G6qLzgHBYC(tVkAV83oC58r<f^v^!64(s@u7'
'$+Daj*$S(Dh0RQ}BAsv}Nl-45`8=2-+z1j>fMx+`fFTL0`xP|&T94x6JmEIdyL7VVHY#H@IN)qf@Tb8|GS8+knrX0tYC)9)XaV$=-*)F&3D1B1'
't04I1o5d=<43gsVCcvj+g3{pa+jjvpyo5m%!DV)nE~$EPk->-xn5(N40I%|VP9x@WFt8Mgmcc457Fk&WfWQ6~bbMQ6Rf_dRce89>VY5ZDx)Sp-'
'zlW~FF#PqeW<|aT;&`^M)<qh}K?cJwDyY58E1LWgVU$nH`{g9d)$hxR`g)a=SFpJ1ceYBV(;_WP_2eHV!Y6*r^Gm>xOZ9t^R9E^(wn%X`blqHk'
'EmHNhyn<Dk>!11LI<552^#!c%!~~(d*Wau3uDXRO(GQ?kF&`^{1_mQ1<0An_ZvZR+TfR76-bcY}SmoqmF6*v~IaKWdZkMSn2haYIXUhn`&<mVX'
'DxG98@0&C(Cs*krkyYA=F9GZ;M#%J2`p<P*R*~}<aa2Al%e1nKy-LdTEWKQ$OZmh>Z}Id`h#J|%c=9PN^Z89WeVg2;g?M&0yF0njO?{VVlQaiJ'
'6xBNQA!UUe&dD^Zs9Z3Cjaww~EHC28JX=L_8^tLUiE;DP+gauFY;rH#;$JZ=?qXFznI$e|1c*T(Q#*N)C38`&-rPUeuvshBUbZY(uqQMCk$P05'
'%V}ERY9YeHCK^f1?OmqTTYA<Gb<<%7%YJwI=Hx?s^8M$N_h+Xc-k(9Ohp@L;9feU4_HNUQ1$|lMH~jZn{5|5o$xTLotCAld!k5jje|7rq_)jOb'
'R{ycOq*DKYolaj?%S+kKioRTCGx~e6SYb2A-@ZAm19zJU7*@{&ED8OE5z^n;Jee?X@E2M+`~2bKUq5_$L)94{!sH@fS6C>S0llFwmr1syFY_{|'
'zso88DN_7%kxr6zDZZvl{+l8EqQ#4}x)m=b`Fc)uaWV1lRdPc$vt>n1FLG{lk(UHSzMS9V-|8x<RD(dNvx1+d3-On$!Ix5e$x8m4%LhfizP!S('
'Ys3cp0$6{q8oi|+@GpaUE8Dz<J#{6%%=t-H@dG)U`+P0op~2;aY8GD6IP+U>;+`9r&uP?*99h*8z$Q!4jQmT*MeDRKzy8$*k@MHTn#_~36qF!6'
'DT=&ke_Ah*{LzojkO5dFWyuw$=`4ts`E9!sbiWSDst_fL6qv@62xkwT&hSe+{N3NWzgu*FH~sv(KM#NRZuq;i{{#@I%;kbgbUH?}i{vgY0nAAn'
'U)+N*z<t1?+I>K=_Orv_NigUi9*IA=O{;cZ;#N$TH|_BFG=BZz-Nz4SC-J-E@8h!%-+p?162JP}=aVx40U&G-UWmF+o;>SxEI^9&vJJa<0Vp~o'
'<~)W9=L|W_-w->C0-*Hie28uFTqqvX!}**8Ak&ZhK)_!W_u?CVh!QP;&?GUED;#zy$q)b#^D6-cf+j0M>GiyV%~mTVfmV<TAUj++y_=+~DmbM+'
'XnlvCR*i8O^MG?Ug%(G~>yM`&Pw07C6!qr<gpmRTc-rK8^bPF)Q~;FCM7KR&+6l;E!EZ)E-+*hHR!KI;(Q--dt~X7wa64RAv+he+(G-VMj=~}Z'
'syzYXJMJF!`~Bg$F~TGRHf-VgEPS8|+5|J`B!$&};Qq$l!3(Ut2|E_<F%Sl-NLepltuKJX#K6_t@SjGD_xUnay9n6*6ow*Me;e@-espllWGmVY'
'VqUmxj|71<009he4v6zrRS_k}zrYECyr6fYG?1xk2|RVUb<p7lAcq0N8~pk6=a22PPVnugw;=cyAS+G~Nz$ntIJOBOJ3aJ3m<4v%=@krl*b6o6'
'RE5AL@%l<oVfCX&-=_HjIl^WKuy1884I+-%1=HSJLf(W81uO|9;KlgpeB3_|fTjlc6QN@f5eWvC&X%*hjon0Ys4Q$2!O_vPhz<lJQRD2x>%YWj'
'pFf=(zw2zQoh6%0GD1g?k@3y;0nWnL$N*?iLTrJ*iyXyvl;UVM0Wa|nU?Y<aXq6kq0g1ulnAEt9KC=}}f)c@d*(w%a+v0D>C_96ay{rU*DueS@'
'isR=8^E_W&B$I37Nw&O>`TV4wL340W0~}XR7O-8iNm5q&(V7IRY?i@((_J^M|3?kUGGBIq|5hkri^*xk3AI7XWR}Ko>`4)A)z*xFuFMjEEiTuy'
'S$0PcVV*!V0-SVs8yA7MDEzn8=B15`#{<~DR6=eFwGl)`v3qNx;C}@*Iqts#S0*#2NWo#lZV@Q?NRR(2Ieh-2jc^WW0M6j1*=0&QT%6i^qJkI&'
'N-1;(>gB6+*$#^fm>s}tXdq1%COd-#np~}y*Em$zV?}$OEH0+WP!t6cpG?~Z%kD(MMHqG*6ku?A>lG5!HZ^EdU{T;Iz2k3fvDkDT4QR;I{B}9d'
'lW9!PtTivIY$@~zI2TdyBtm(D<qhKNumni5VM6eaOTuk3BhR})jPIIQ>|R8G==mKU`wKX<!LfWw^J&omIjXWo1SabO4kr~e7Jf9(78yQJW5fO9'
'5wN&4&mAo_0n$6h2e0QDwAz+@eDJd0>qh_=Dl%ZFv-Y4A3>}9#p$ZO8#E1_?vb;>&FWklUa!)auj$PCMk-O<JTD8MJoqPuFB5Fm!6`)xPlE6dw'
'EzqLw@g+PBhe3Gu=l-C3eA*TBelY0u!wrEAL~E5V%hUl*>|yy9`b(;H8Iz2H%<Wl!5CzZrhxpGk{O9-h&&!Rc8Cnsbl%x+E=_UoN$b;IUED(15'
'4FD`iNMx~wZkiD6lZi@l@o>*WqrqDV8w1n?frV$N#AvHomrym;goAmuT&Mfb@YkaN<hO=FdfBKmLL?TjjZnS`9(-MFe8U8>a-I|<r1@8gR1C|i'
'3;g5~&TpP3U_`1FRM6qUL_X*QU-Q=7e_-~wkiMi-Rx<)9v+Qz>{oroqhJ6rKd6l39-S>@GQAAWz;l5g219G(aM>+ajtW%J#@8A#==hyVp(*x~G'
'vn^&8w}M#!`B<%+#>_k0VjpVi6J#WUR++403-a?|pN;}pA&Y<bKCo6?e*3b>&o8q)^24*#Bb99<<c!#YBYMF3oaaFFm-lLqmjP)U6#GDxB>`(m'
'Ns48D3CA1-&j-$Vd@sIPu!6GuX{=MW@-uSJZG3A_o+VM32nhp71jJrZNk8qUk78j?UFM3sZ5<Bk<J(Rd$?MyP!0CbEcZ~P;G2y4O#V=-uyV>B+'
'&Hf(G_Ui0Tjgf#OjW<tR+o<{zn9nYzqj&7?6W5nhG!LWsi45jT7>l2k5GN56xQmwv9HT}WSVFTzvljz07*EsE{4z@<KU<m2FT`Zjy(p^<#8IRz'
'P9%Inm1;BOHB#-H=Wx_bs|s5k1%{IM-N~EN<M{1|*T-+AhWE|sC)72CN~NX_<D_6Qn%YP2#QIslBbY0?l!Z3ZCF&ohUj;b#4G?rR<Wg;5cHQZv'
'cPrSY>ymY7toA`|U^c1DEx=PFn=mG7?-q=KqwWQFUPo#Txjoj~%wc=oq;ZuKpq?VALTPt)Oqu~<uvnK>Z~;h)m(Iu=$_!Y<_6b9s3HON9v5kuu'
'm)VbS9Pk}p_gX<Qg>sW@yBDw(4P3FNT&U7{dRZh3B{H>fcyTsul9*gksID}q^6PYIWA^8hw<mx4bo?%U_2F~;`NLmM-fzKbs=s40>J3VK)+9qt'
'z+UQrqkgaNKlK};PjU+qygaZY3!Aa#dU$Y<tg@b*%U)hw9$e(r1C3;J042ZSTQW2YS)e2&CK<#a@onP~wzhKDRdSEZWinSZLoka_(}cVuS)3U!'
'tNj6<hfRl{sG<%RW^KGtHNOtkZc>%@;KHB9+yv81Du)Rz#db2g=2>>tUR$ObTQKMQK>@Z2Eex*d){+Sc*phS$(RNt3O(2S6Yyg9vlk`ZXdM4?2'
'*-s<C@KhEPVL*<{jMVoa5JZU-yG;P#fycN#@XlwwiaS+UH=P@;D_S8*M6HiHHEF8r@2uQ)*YiFXJWSBRa=lu?cAPrI&S+aRY_hZB8%Du<j0K$n'
'qZt#$J6_(O3n+wb7TzL~Jx-6Vy<xZT^U$VS;^~eN$Tp4Rod#aqS<;QIZf2}(#KJDwGSGDle(9E1a0miwRAgx>ZrK+}HJYvq8hnJ@W+)C(*>alR'
'MR>BA#CQP|-DXob0bixr<y93GF&Uo4XtyQ`;T1!~W@+N_RkB>dDpA)t*<K*c5jW60r=4@-kTrmVm;M#<((&Ah=id7+*d+-}1Ge3xQ*K+|mIASM'
'Zc97nnL`X5YR{l1oEo4hLE>m?Orj?XGetuNU>)7t-zD=kFIyCRh6xj239SF;?8Eyv1O{6c?02}Ai?8%~*^;P&K<Wiv-T}b1Lp`q`UjOOwh-VD^'
'SX>O>fKZY{v1NK!wcB(zXk8MDcJN{h9?4^v#0Y3>Gzvm$B<#>k36i{?W<R-EYQ$<5k9l&z{s`wR$yf-09aDM61VEg5*CpMe@$eZOha@~0c&HYB'
'sQNe26gLeEW#N#Io?DOz3kw7?)e9kQE>WMr;^O?+Q=YeBy@Gh|sX>KEQ!#;RMisq#L#x7URoVs3J#QC<X^nZkhQURi&)f9FDnd_G_w=i=`;Df;'
'fI&qvskn8E4kHyw3gEJ|b0f7h`oS()bT%Wa&^o!s$2Dq=4|ncYYS2Tg>m(XWB3-;^!i+?67C!Khde3GXdR*~SH8k}ivxz8?da`%wvUS>kzZ2C*'
'TaG^co(##BZ8cfM<=_R0nAnKPT1|3{rbbYgt&7yodlJ0t1W%Dx*Vk7r7j70!vE^Dyn`B_CbmmT{?t-##v1485$?|fYT&8?$%@>w6iHq!(Y%VR?'
'bQq#$krW0+N;aT*ao<7>YCb2s<ylfzY0(9AC|7BLR@K+=2ME_ij&^TTR*jTBG8@Jg(eAn&diw<IlEjv3!66uq9Mh6-C)>|pb{E;QJ$TvQX2kXR'
')A9SW*Pl*5em?#1K7Re@Z{PnVK0A30Jmrj5_^=<c5tm7Qr37f1f|GqCmt!SJb3q&w9TGyV7tUuhJoqc+aG%uS?EwRiw#&Uqn#~oq08o)VaP!>C'
'FAOG(Q(=|X%O7+_=k|%iX#&|oZ{Qp-E`YvY>qkD4V&fj5H;V(J$H?g>;yE@T5elXC6753TWP`rE^l0>&WHurS<$c*>3wz{FxM9gMla1W;<`KJA'
'kXZW8-FNifK=L2bp6{~heY#jx_u;vt@y50LgWTcHR6c<%P8;1;gwqaETjj4}k65S^`t|1MWgsocYz!b&vn4n&p=dS{qg8V|Mf*6I)R9MwL(IfB'
'5HF1u9W)^EV??5R8{fOCbu^R;Y5l`<B@k)>h2KOZ_~E=b2*ar2YcgF<y$z;h0{n>Pct<9y71X45Y>|at*EuIUh&YtCMpQNVq#d0t)(d&Pb@b9p'
'nNib?N|5cumL-O1Rv2`KAqt%WduzrgK!t^MNBfA$r+~PL3PGntrO<-IeulL=nslOi2N7`$BO~g<U5$bPoi>BEL5-lu^A7Bi3|eY)!EP|<u%xRg'
'B)#dJ8-v}=DSnz@!q~uFY0AyVIUZpJhctvRU9K18ghYBRi!=*8kY2t65JdRb12r}5A5Ax8bZ^{%SjRH_c*q=nYVyw&NSW~k9vAtbI?pf3kI6Z{'
'N-4?SD{4$z<s&ioH(ETpp(ANIN#^P3=w*M{J*3C~<T6L13DU5*gcCo1=zAWb2`<_JE-O+g^8|XtBKfJ`?~%C~;XRsw<8>B2u~ovJ+Y%n1Bo7ZE'
'&I(A{1<EXKHAPYI3}=HBXhS_dQ=cf#vnTWialLj{#SgoIfH+YsfhxBRJ5DiJF?1jlzv~^&0O@~Mh6euO-k#V5Oros1GMUWNWEqp1u6={2Ys22$'
'wY@LWUb((#7vcE-B;6nT-9NO>pOV%%3aF|yPn1gJuoZel(^bXXW{X5uqmRHK`+Wx}Yy=G|(H{lxOmU($<;T`pqpeih)&ZIMAzlaUs%B3h_BtL4'
'@Xn2iyiV__%{G=0s(NP>FEu4cbAK@on!3_(12Cx+S{Wp_e9zdM0aXA|0~g@9`=77`=i)CeMfd#SFv6XQ{i~HQFiKtmSW6C#+rl1~!Y#qyglVaE'
'!Ku9x@Y|@%59Lmw@^(LSiEm{GO|j%mHA<CY`nifR?KJ0cwv3q-Y(*;j(37I?1%2PeU`5|!l7CD=NFfo+9r&{@cYMwL(!{ytY=P&G2$P9Zx3J*('
'ZCp<B;-?IazDZN#be>KsG6C?;ooI^Mu(_bDS?7ZQy>ZV(%-upV++upn#un>Xvne|`cN=4L!=n5xTgsObVb7&{ep#Zg9bb)enT{!$z@R{apb?6}'
't=7v)wU&pYa=mDVR=Of|deBUXw^!|U_;2{~H~7ymY#5J{sg^Tf!Asz(jKw%KjS%Ay6_LqBhTekVxxXpW=fSxLMgZ*?V48J-`LxM@h%rD4M!J77'
'&>BEERb*38e?fwp3qQ$g{oe1N+gyUUA&QL8Nf_icLHXO9>nBk0=s7&p%|^kI!7gS%>Chj5oFup11JT)S=v)^>+Y|=0DIDii=|RKJpQA<xPYH?W'
'SM(!q(`lpg8C%`v;4^*JMpPs-XR1EdiqCT??@E>Taw=M?m5kK=4-`_`4nUV$NvW>5NiuR979u^I-+g-lz5+5i@bg1G0M(YJ4!Ze2j$|WGVK9fL'
'YXN%(6Ve-=!fjK|90}4Ro&kWP6dl8W3&5#$BrpG&DuxoToja=XicBJY&!-DH3KAGAp-V6Ds_!ZX%F8G`81xK-q6rXcx)qezQ>|C?bSy}#`Nuj?'
'P}Ox&=6DRFf#~OKyCUBHQOrEd&vS?0sjg$a`ZNQg>Qj1xBaFm6cRCF%DBj!Vx{=*@<YO+fAE#`m)`@_Ik?9bznPwgL^z-LN;bodmiPd-&SOQZH'
'sIt`w@S|(02i*YGYwJ-(0mxY+cKiTJbBn_F?f8(k#7$b1sjCfAT<fuGX$p%BkE`pLp~WM~j^n{OqJEAJzUT!es^ZtL<>LW6htUs<J9y3+@|{N!'
'Vf2oqTD5{5@pOCIbF_2K4BhM;=0nf>c*u>P*Jrkei`4clOaI(l)Ae%6_Emjzx0!UonSFQ_ONszr32O;k&FOOu9a8l9<9-Bwb!FGJ3DkwXI!({a'
';U_^JHaB*t22KJD8&$R4ac7suE0?yJ78;}1^iX#$=3+$uY8ojF5M#l1LwE1WWq{3V45X`J_n-+wTtkyDBIt^wDa``YF+Bps_zgO5YST?+Hi|KH'
'c+)+=Sj;Efjde9D02dQ8ThHgQI7@nus8K+NPcAJ_c_==s6Q5)<7exkw%Tqj(qIz3490f0w-4igOSek1Kl4OR`Fo_WADM;k#kx500Kqtk0e3O;1'
'Y$duehxu$4(!oN4fHrL0%N&Dgl=M;pYOElXF7D0Kvn(?nJ~MZ%4Y5R88xza{cob38)rkJ}D03K35GxLQ&kga&UF&Wzz%!Dk1&LKo4Q}fQ8+F@4'
'KK0Ds82NVNOpn492#pw~2F^#~tE*U|{Ul`*SR<f=F_8BdO^ESsqROG7uhK=jOlKMN@h-XMTQX6F;}WgNW^MxVbV^2Y0Dr)4=U4=buZnDep?a1{'
'QIHa>=j%p`)}wy!#i4KRZiGZ%q_c7~4i)5jBZyt-)6Ss*W(*`Eo?v7as9vssTBb!jLkD?`qmhtytE|;FFw9m9xJ&hQFgPR}MrJ2*6wlPp9CEY|'
'qT*)<vf_skw!Sl-4PTu5a0WjDjqx_uhrM*#dD&EXD+d3esl693y%>xi1B);6s=^p1{(^TS+-&}L_werdvxb*sG%*@9R%ROsXyNGh4JCXVC%wbN'
'UgA^Y%8%MQ$J1$$1f=KcN)0gA&S7KV1>bFQFC_2lG+oK9BXE?EO5$e#G&O_si+2~{$J$BHUo@?wvzr##bOjhx?YW)sQ`=IFTL?d$+OFaUS9yLd'
'{EPX)CC6NnEXN4J<Zv9tw5+2WiWC!8STJw(C{$57yqafKy9nF=_Q%n<_nUt^@BC1rf><@$QT9lp1ykaq&~78$8%ocppH;onSxvQN8!8*4riRAD'
'!E>(9?NVJmA7ZprgVg`bXcsxobu4E~u~>)J$oI`u#+cv+B6AIRFbWZXCY)oE(!4(#sm7q!NNI55M%d|Y!U4>y39@7f+z0!D-8;uyCA2MV9>)DV'
'9fcS=VOcd5s>y!Fj@dRx?v69630Ah}zL8(mu}UBvvYG**NR2a%f|tGiHtgYeDrrFxqxif6oH}$J)j(>TUZxi256%I9U-r*^1!Ggm<du>leJuC#'
'p+omf@blsG1T7)fjmyj^ZI0uyxNXhB)8MdoxE0m8_-<q;HWJU%StIBM&ZDLwneV`N5kcci5T1el0+i>-3G3A(aGtEm8}JvEWrV&4!u~cm4ut8A'
'ZIHs=ICNgbP$<>>J`mmnLMBV60e8?fTT`8(o)z^&In;oKK`WCYU-bfMMuvV!tug>Y6Yw!BfwPU-62@+s@sT#DvWoO==~p6pWwyjvg3?5N0_`u#'
'o~}(T#$0Q(jgwg?J#|3{oQnC4MMsQAogFV2l~LgTQ~bpk*ME05V}QEvenUj$dXLy<<?Jp_Z%B*65||8c*>K>W-T7n>ip5k4<3LB3;3#KQe-eje'
'hiyiTTfmGGqXQ<@h+66iRn1L4UoX;;y1kg)_4tui&~VSltjPM-z8`n4RSfK;SsvNS64*9}tS=2!P+uSvteiKI3=uh2(#zhOeX?8Ct;mfvl3$E>'
'7YT+%6}q93{9+WaCwyf8G|K6@AL*~gdnwV!!p~F9wfn-T!%nbyfk%TBu+U!Z@Ed{891eJmQz*h+q9s0u(2`ByzNS-2aEl~VUftA=KC|#lDuRU-'
't2StRaA?A?cR1UW^%hR#<(WWis3|a9swvSKVRIm#A`?LMLxwL=eW_o0bzmE_OSv06X5dM3(R?F@E(22G)jIT%RdI|8^qHdTzU##&k=C-%UF6Wk'
'aVMaYb#j`|dZKL|r@`b}-_bzOjy}6wB6lklk{%F-l?P|vd>?W2u{oM*-UqOwiyUL13Apm5*&-^=1)E#L_7BdM$%<}gbkJyVeOm%*Bp9rDu@=#n'
'QQ;3gna!bt(Phe^sDVUWrf9`Y-pbgKWc4B_qItlEa#}m(Oi66dC`^sR#he-LQT%M>uGhA!i?`GDV%5g4BX)surCd~J+uaKPR9ZaG0Z|>N31Q^6'
'A&hZMk?<qvU)x;s-h945KR%o*zt#EMx9{R_KfPrO1E<*Ax5w{}<Fk`b-<^EYg_K>Aii~&=+K|$8JQ_?6QN9(>cOTxIyj6h~yON?9FiAN69Zl)7'
'$s(<;@~HtDhX|QM^96_&zzDMmkWkWf+;cEibZs%f4a4vl>wr|qZdq1crgs<$7v39@33x2&BCE<^y%e5E!rWPT6d1GNAM)|rw;%o*zd8AQ^6A~_'
'`_r?}r?2BrC$B%ia{P@v`!K#q*paqYbl!6aGZ{fv#hSFC_sSp_KO1Y!a3FjJK!2vRDBJdyG3XqJqGk&-gR8(pG(|Sv+H^@$$7b6HG2b7E3jtBe'
'mR%8WHTcuVZv$j7c)x;D$X}An%Q=VKmo-l*egRQ#t`B4(UsYW+*5RPWrfdA4zhP|Z1)?oObY1|JDHXLtPLYtsPcgZ|Ii|P}a<Q3iMYf*LkU5ZS'
'$tdv2kM}Tn%#VpuEzxysE+W@=F@|$M(IpG^^;@mybE~xk+9-e+zIzLchEf&Ch4*}DS)_}+xbImLX@F5P+LdA8aZnD|q{#0ECaC!D>neYTi2Y4o'
'yawK$%-_C?=;>!Pbjp6j*x|C3ePSKa2JktCo1M#5*QdCPwSfD;h7z}$5+`c@lK5ddolWgX1RkvLg(9F?;~PoY&K_g&G>(w6tA-U=Mob~VW=&}i'
'hXK1Pg~4XablR4kN3ck!tmPn3<aK9-9>xzjcVFKE(v+@&5=BCv!r=l=7K>zs(U%hFH{4jGrFrg6*3$$dWT@F~yG8>>N8^wx#Z&V0#D2IDvT_)_'
'Fi>P(i%7U{GtUk!C>s~^?T6zx@zJZ(&x#D%)Ur+=UFBGP=WN|0uYlqoFQ>22Q}i{F2$Mv4hxyNS0tD+vA|x@B6&h~6sfLp_wU#a8ql>Kag~gJe'
'i{%BCq451{KpC3ak+nKo*PLY(YZ-QRMS~+DdlW8bM~#;%vOJs^;FP_F?zn2xcs%ibj7jaSau7>Q{E1jJwrlNY1kY`gAv}dp4v)G3X&{3D88qiB'
'IQihCCA$T{o}FYRoma>#UfaUshed>um#S{r^G}`_<LsEog>3vXU6QYd3Ij#T0)r)c1Tq2!cq}o!Q82EL{=vU!VUf?#`yedufi5>*@kxsiDA|C('
'RO)Z}nxk`zd6S;oZ2#B`J_$)_nd-Y~GYL<)rEM0rh0!XmGP>$VLN%Arui%UllJpvy@#&k~z;rZKYEp3KIhBJPy|ET<RY%c|Mik5RHdYqMs84@B'
'INatwirnS+`}qCIUpe><JB_~Vlh<hNAaa$INmUhX#da`H1srhiV>oZ2@yTK}Cl7GXW*{#;GS5=Ad!D)ti!D*DN#{15#sZ6x9`bZ;qiQTc$9P_+'
'@c1jQ^35NhOwNKB5#`rFwL*qL6dQt->CuaCy(;JZ*<)_n6x>#}tZR;q?#}(Sl>aXGAour{jr8P+98Gf>8^1>O&n@+)dCajtyrC84$1?#QRxdxE'
'dD(MDA3ry$um4VSBjF~gl!k=rd<7~-D_^DK?%;gr!4c^cq&A|6gQ6OZg7P|B#T1V|nafcL8&!uL*7vo^KO)~&>+D#orZv)7x?{rKd3CJ`8^*6E'
'(zn;lIo)%D>T;kAqDJ~8#dt|cz%BzWGmZi|k@i{H3eQF7RySgNDiXNWA6NC07va>TH_7LyJ#E)%|GqLk4O^F`BTGC<fKRq<eUESOjoJfuR)#92'
'*jVJCp@08%>W1U;np6rZ@IzjRBzmQDdnfN5l}iSCI>Lj}(nEltQrri4%4fxpfb;+jN5o(_p#xs)VCIN<Kq+W)_AsK`1Kk=4L1m{GoZ%rDl?I&0'
'MaQ5!e*XsWCY{TyfI>-$GSMter+9ug<_)GE?Y}XF5-Ls8q*8WCXu@P`w=^RLF_#sw(<T*zw%F$}Vt_gNVjTD=gN@(O5I=OiRH6J}M4uR-SB`_3'
'zYR!5fG4hqcScu;<D4TAwM95h>mt>;Gf2vlPkVNQ0v7?Engs2R%h^x<^P!H!C)CZP=)uT+q-PVaSqB<Zu8o)?^@&bhfek;faYO5hD3tz>qP0f^'
'SGC$PL)m<#WAIrrv(WzsoVBkZCcIa(Zm77MqNB^(gps#}Cgr-nF+?HS3^*~)Kv5u8Hx}qm#CC0%;7lg>YZm*ARUK0Yh`bVX@x+4%e7UgWHbJ!O'
'lbh~V(iNiV5C$g#w2*+42Y=f{;#mwl(0aqE(WuC_wuJeIuevl0+1Aj8`~fZPzBt0ExkpEZs_jE^NnuBN<lW#wELBgP_1HOs9?}NDIjyhk#*3w*'
'i%0C1cLFG_fvop6B#ZknQoi8qHzYdq4YlC|jeHfPz+}|`mlV}ijh!;0=1UItLiP>~+c8<*x7mGyXV0`*+;aw-HiK`LPVfrFGc)WaPsTY?lMaL%'
'y=cj(*$2F!L7W}ad?BOonLS5=HzuDa`kpa2>fm=bk-;aLx4z_}GI$M=q7tk2xMDk7PUh<=8tQ|DDa8LkK3o@bA-QfjJk&oRgre<#Qcki=HYrTH'
'MySVm)EUbdLCK@`)u+KobB$RjIq`$|vH?;c8}eoraic}ZTW?DuA>k!bZ2ouU!84&A6OSSVt)~l*kkPg!LA*c|j2iY*!&xx=Vfp&Qr%xxZG1OM@'
'>Eu7YJ^gg@rWgDzUl(*kljUojHAhNZdKkSXZBzJ3vd@V~kSN7TnJQUmo`{>Sk_z@Y0BkCt8k_zoQjf8)CQTz0voyVM(1L}n+EtixkY%U2h_br?'
'p4HgR3`9bbn-*z-iL22d4)~)z{BO{VgEqWvg~G<z$P_UWc2gJZiW5*^8jvz|bOQMV5U>}#MhhVf-v(lBvaE!xEmZI36f||Cre+xQ(ZP=UVeeP;'
'6UH{yiy)to>5X*DQdTBq^HO}q^r<9RzWVUyZ}N~U{h%+>8Ttq+QixJiT2y6Z#2c4|oBMdpcoR?#Z_24H1RJqWe^`E}x0RHv$?Z#QpYaf50-gs?'
'8~4D+8m6}w$vs9iuL_X&6LJpyVHvWMJPM}c;Q(_^dL$jodA80S6fF>YiIV09&rH%F_P=DMUrAQVW9#g^+OLbzTI1~v<%=J(L_tSWuHn7$<IpmM'
'Jsfl){QKYkUgH+BB^9IQ|3InszYP(f&cFRoJ~0E;pwDk|S&pFaX=g_}P!(GGUuJcqiuRDj!xX(WR%byJh8s&0ILc#t54|TpY&?*h*~8|#xGm{h'
'@3P3(t9HMGhJhTY$2mjQJiqOFmO6zOi<Wk1T)<c-c&GGqh>KTyN4YR$*AyI1yU*`;R1+ynjlguO2@ebMIOGbk#1{m(Sqd>`fW+O2IXx;pDxRM_'
'XQwT<k_ZLR<5i;5k@K&ZQ_}myEqpDKvZ#`67t^*)m~7&UFW5XZqk(L_wT%g#*zs(mgNrb1qO25wm15AT`A2RBGSPVW9BA9dQ_nOeuvUqV;5r@('
'&rL-1#w8HV2^`d5R1D~ZMEv~XAIJevZZPF7DIQZ_yQpG==0WNV9Ll(zScV~RrIbS~7mGRC<&@C)yvui)90-|2j%hZVf%rgPQ-U2D;)A6(lJt&3'
'HtLhGF7VsnGdi4dGGC<A6nG4SJkwpENL_@7cM?s`5F%Czl>zJ3K#D<AEVIeK5B`p;{`YVY{#`JmGEJo_Of)9bk&*>bD#Xg31Nt~*rARC)(cOJm'
'UiAkDN!Aq>#s5AC?IHcgEd~@Kv5D*#*;EV_EXh2fh}RU}Q(yq)SZ9HS`htCQUE+M3O(sf%iQ+;pu;fLl;F8|Fw`d}{l;g!Ka$&UAv(Vc>ZRUa;'
'B;7Iw4mm!li|#apk(?u^%6Xm98WZXkM$U#`&&7@t#*(J$klaZ2h1Rz_w{&eB`Wi|_p5vO18o<kQj5nlO(TYiX1Jn(`Apy2}m3MK~V4YxpA?ktH'
'P3Hxn#wU5yW_sJyT((qB?qA9h+);SJFKO`GpdfRns%fSPpoZzyQ>>I{WAnEoQ8q89?8Jj{zCL{II*mnRAbWl`+t8xD4$}YBK)KrVLl^Z)ba)50'
'(>!_pY+Uv%8DVElpOJ<QO7%H-tyTJ5dfC=qobT7U?Lxe>cc-|~z5B-L-!BBAo?LrNhay+A8`#eV!6<JB@wH$1rJJ^q2STf&n#b#OM!{%|>oMMo'
'F@|j|>uTQ+y=O=%%brzI6H7MH=t5XsWgIW=pKG!TO!$}bi-fNg1&9YmGPy>YL)4+73<eZWbe=6q#e0#UuGdxZhRSt{pdjz5AY6mMFWncRX>!fc'
'Fv;g46hN&><!41rg5;C3b@P<|vg_4)rN6Tq7J$N?3wOv`v*eLQO)6i=K>23{q0m#Qxsr2ZwzJu})eYMDOKoyR>?6)_NjXAnM8VvDi_&|8YbM}e'
';wkTqL}S}_h9Y|nw+}SW0ws6(-cnf;dla5*DQ}{Nz=$oS<%;iee%*L4Z2slMv?kIMUfE{(By%?3rZ@Wy8A<?ed_w(xTU<{A|F?v-Hi|u|-_dir'
'5SR$lFvequxI6Trd8_1&e@HU~Xa&;HdO-ndVER><WE#Slf1O7nM!7bCfD&l-vEdCz5j`>)uy{N?JmMIXG$a;4k6xNANw}n`6H3jeDD;^ylIoYm'
'e5n?;w85YAmAKcxPVf0LJ9O!sb^eIh&oR;-i{yo0<l?&}D@Km~HmWzSJqQCWHAbF`o`?;iGk%CWF;cE%QO)U)FY0YUx~2@RUsOw7J?o`Z{zZKu'
'hG5<If)l(!BXUiJ+;-JXlo_K7o77>v*z?<Hv}ALRTzaP{MY-F!G&taZb-my{*^me&@&AHtU6LgbaLg*hfP(V!oi2FLE0AZ?Z5Gh^8aW?yjOXfz'
'wH7+!4BhV;mFgM;7az6BEg7A3@c{3)y8ZC}?cYQ)6?ws+Fj!q{)R#|ofJK&Mox!ogG5rOeJ<F>U&gytbPe>R3-s3B;f{9ZuT*|PCfY9dOg)r!U'
'np^-eSp%G|;@!Eqz_PqXT`2`MD(My1e_Sm@D9ephT+IZeq9E*oq1|J6fFhfT!y@06!mweqZ*sxAX@X?10TE50*4<yRLE|1s#%-e|7lH)he4^Kq'
'ivW&eo%Iq!8FQ5Sh%^*Lx-iUZ?jb|Yd{S~vwOz%PkMQxvUSW%U#lH_yV=v1qAi7v_yKpXAN+3Yx>>kJFyWpW%%4$D%&Ar~K&lX*2-I1js0x135'
'k%XYGiTzK}Y%pj~_-5)JZwqllUKT>}LZdw3i>gjd0oi)so3C&fs!J^~;pLh*Kp2UD+(QhWLP5NT7{geISRxBs)N<N2r7B<4xJ@r`s)Wb$GM87>'
'GNe^xg+jK(l}OQBeI9E-IF7NQ<7yb#CXfiR4OO8!^adv@WR$_7nlZpRo_w7gVmagp$mR}+!5tVUpaXsUIS_K`0SNa8>eO?9N1g+@XAkJG<3|cz'
'MeT6h_5THEh`C!U87?3IAp*n#z8uBYnK<Cq2Bf|~Cn1qFg3@w`>8<yv7BkWTD0u#fsrG`ur0Gge9>zJxoJ}gO7{<OUt>|Lv>K6fOBxMQdAJ-Qp'
'$PY*_XWEe)NbNkoTzh~;l9o)dpPMu|`{sLq0%+Ira(+MLq0cBPbb{1>ww_CT#}!YQi)o>OS=uVO$GkZ8u1;s@gDXW4$%zmJ#Iy?Z(ozZ|6l`jc'
'#%e^GXazBsuQ(>|E#f3hCJe-Af&_pXFthN%5%)GfEbC=hY!yO0TNK_^ZGPJE)p&(5Dd5Cv#i-cOO)XF(;{^wufH$8x#;i>lje>UcY+s+>sVPMT'
'uu)ab0@<rURa{@E42I_rZ)**o(zUl~F6GvC4Z}8HiJb9(%P9zOtN9vM7sE)D!NnSOXtkws%q7%Nn>b=t2MZlCrC93601mSy>>9l0Uk*)%Q-?I!'
'0-`J|=t45r8iq_SBnjv${=qsO=cG6si@4*%e)m~;zKL97-m<!qHhLl5YDifq9QK{wRi`jpOCCnJt8JlcYG|ke@`EuC!_9e};hYLyLqvgYQ$(|+'
'>DhZv81hwso~QxZBILNE*9aiG4VeaJ4YUZIbAc*6@4ghtF;=OvbVKn|2v3v&aDw3}v@M51j8`Y0|9W!r-b}1stbB(fwhk8}8f~E(!9-?|$PjC6'
'vVuSYmI0L2ZEg}rOL(o1?-*odj@~_43%MBPZid#}IZY|aOyat&qG1WklV0%3uy83^*t$YIr(irHeRQ43eG(@V5vf@ndFs1Ro?&8HWK)M<81%TO'
'L<|#hFwIIj^B<JA*(yB{wg+flyDBog*cU_OxK|!x^Ulha3ji9oHbPM#_%*t&v|CXNaDCY7c**c*dFDW@Rzb$>M#m2=;vuae=bI3TG{)lTc_%vT'
'L@f|tY6b9_c#RvWHM}J~Qw#AG;A(YP197Voy!h{zge^`@D{)9>KfYmo@}#XQ?NM3Ell`acY0wV8J3agM_-*j!<m~iM?}LxWXJ<oK+2FAbMY?Jb'
'Til29d~)qI>k0)D=s+^s`i)MZGWEB-v1M$L$EG5^EwYNX#=>WJ6~5shI9h4pJ**+b$O$PO%vDnH35orFWw@Ez1O<Cp$nyeXAeN{8y_4{TyAUW*'
'>buG=uewHN6}87X?lwq@SY9YFgyijL3ADPAJa;%C(c_bvL|avar0?V82aYv-r2RomV*?c>O1cH8_&0`PL+J3g_>^4DV(Dr9#UUSL8~fQ4NyAj('
'+`$UmX3eFXHlMYtZQ1(hCY7U^o0CF*EEl$V1egYQwMb37_daY7Yah84%<w;j;x)^u%nkuiZ4cP9zkT{l*xFdJMW(h@B9KCWcjFBm;$mELXhi(%'
'(b-pyuaz7WMMYbax!ZOFCbn2QJZi95YFuqQuhABjx^JDujh5Vt_4}@0Y+ZK*R}pcnOlKz6sF^T~I0T5@xVxg^`{sysI>gh?<?Ffq{2bH&GZ2@?'
'P(V_1=b8vwaV0p|OwB27TWkq|7|l_=ULC*w%gOsUbeRn+kYt(+t6(h{T``}}lSL8+#)9ZXXyHM!$__99L8BjsvT{~|@GwwFV-$Y;sD+0CZ0}P9'
'2M-@)v5kJY<;oI8uhBzD>;-dm(z0^=v<}_N7iqG97s3N1WE;D%76MSXl#%f#-v?2B8EgS6jJJiRaMyEl-sB@jC<Nj$90f;|*&L+JCeM}zuP3<<'
'J=)SS3<;I8)w*=e!Q@AKXR4}Xq1qAlCOk+O(ZfbOk|QIYUT<`ZtIRKeB!~kC!XJbUsyxVYGpLu_i8syf@za1+J1U_OKJIl?91k%m(eqt3{75E)'
'!kB0lek}6odLpy{+E%xRnuo{X=Psb+I>u_pJcJYp2--Jx33g-tBm<O$Zi_=Bx(pF(P9!Xf?YIAsj7=(evAk_h{SVr2lPf&0k;Tiy<{X1hf9XDy'
'@+U|$((~EnKEi{xIm$x4Rfk#WunRw@`0QuyD>~2A`SyOI@xNGnnz`oZ`$@sOC18KMUncPG6n1|`DNE<1R^*1mfoOThYcOA?Rz784UTD<-?ThR5'
'o<pgU5&?d(8LtQfj^;}$tUX!fC6{EIJHu$VNSvII*XT`cr+>zjVI3ovDJNOf<7(4Iq(~CIeQ;*(-rHnZ2}gaiM{&73cp>33(|vB4Ogp_VH@+pg'
'FJj9)S&8Jj0+#=6<O3DiWBHA4+@5=(Ob=P6BthQ{aV|S~N|u=~t$EgoWhu#pDB7{p#>N;mG)6mxNy3TOhrvfozuxFtK^JXYp74ND!K3UDv;P4<'
'5WWac5vwQwLKM^r%2y@<K%&BptAz7==`o~VhT-KS+o2KCRTc%;fz2}t*VU~1vKI1K0N`43H(Nj&vn`C_%s0~F7ETUqZ|ifp@-@3k%F<fOhmE;8'
'$Tcn|Txid;iz30jfr}Oe=0-tfxjyM^$=59~S~R8R9ZxRi^Q?@4Mq2A3Gkwd{vUe1RJf|RV-<`fW`4FFc|M}$o+3AP(XV?whf^+D3G!0<E^7P&D'
'pH7T|a^&TDUNxxaoRsf~S%`}g(&BQSU$ny~p@UgC8p1*t3Q{K?qk*sL04Hot^?+RO!J6r-)a4@$sqHDtxCzJQF_mV6l#FasX>DV4EWJv~D;O4r'
'fI58sqRp;g0Ix7QY<DobHpP&FZ&&HvG`qx1ZODE4uGA@MiR{3Y*hRP4en^PpcIB2w`!0&_xGqp|cd`di5yb?CJkR*uEyIjtsHTt!pJXq*Jebk7'
'*1f@Z+}aLL3Tu^#OV#>h)-|r>tqbQMLxN;kt0AAuDBrQb!Fq=n!6Ca?d!aSo)hY^uWwJEUuQo19p=TFxO|s$jGtfwstYM=Pj%x2kekW0kvgSVT'
'_iGuMl#i)o@f=S?RI`#0lXrtKJ{ADik`cS4PN;Fb#BED5f?afPGP~60<{s}q-Xenf8X?jx^Vl}j?5)x}^w~o}7Y&BZIF}z!zCU?;79W2+jsJ4;'
'H(OP2@*E&O+*Ed#TqO7Ywobo4es%n}{To{^OOPT<_Jvjim!gD+-X6YvfA;b1@!6l_<JYfG&dzrBf}PxE*BJ?>{KgiI^t3gz3Y&}9bb9!3pP1Y}'
'HLH&9U?^8-<gbcoBuwQPwQ9$Pec8@D!>j%gO$NpQ)Nb4{`j!x?YLU*IF$4O#C_}&`Su7keWA+GD@tPTEy9#!dkv~0x+(wWoATJtVx5r}Rw}KW+'
'_cby2S*LUE%0ZTsI*$dpq3FpYu+RlK!65who!XJGlcQPl2wgR=c+p^k$fb&oTSP)y+i2c4?|4c6h|h}%0N%ST8DO9hB*)KwzZ0|*f~NIXi6def'
')e-l1Wy=i6EKtgzKzkCOt+;1gVTM;#wJL`Ps1z36^Z*qPIK1D1*gK69U1Rlutm>YE>sn0YixQ$JxB`g^IK=27JYH8<c_9Lj16$m4<Qo(uie!OJ'
'2(wmc<s0ILT_JfKGJ>LJ9}s<6DOVocd_Za{RQxIR)*-34Y6-*lRbT0#dyd+wz89q<m#D%r1FyxFMOa<sRqprn5=i9KYYN8%I7nNYnY#8gqFN+K'
'OQFeR^c-{}FnyaT3oaDxcY;3#gZ@jHz^?aoQv@<o;H1>0bIN-vMX}{|*x?+ebD96tj=o`jWcoU2ieeM!fwkZf8jrL<q!tn|S-G}b-4>D32Ivea'
'4uK25Bj1FJ`{2Q78uRjLm17I#VQh50yk5fK(3%Enf22L*(JyVe3UKB|FH>1eOeLqnWjg#c7NgW1hj&ep3+3$T`pH~0ISU#5fDq6ed^UygHCz+B'
'llPG-+mlBPlD!K};=Ypx7-3FM^Hqu|3O|?!vRTMfEUpbEGEY@ph*V<s$;kW}=>v~&eR2*m_LTDBMV(j1=n19|BVww_Rn$6P7{4O>2oVUed)UY^'
'AATBuo6e^56FH&8pc)$teQeN>p7#};AI%6wC&b3^rc`jkJ$Rs7Gj!PiawSS{0>sl{XxoN>j=<ZQ7zS++{B|hPhB!ecCGfx5#se6_F&`mfmEVN<'
'%cD8D;KKap=i{b!YHiOo&{ftD+Q6KLoxG6U%}lZ<*swu4QG#19T=H5kN49ZX?7<WK6F&}3r#84U(v`<u6*uw+qV%V@xR^A}0sA(+;44Xb_ocK|'
'atk2qVzWNrd~fz;4m)Twx4DZrbE%(?I?H<wllnEhH3_zufY;*I@~u`N*%UlHV>9~TqXzt!qT~Ax<!2%7VgmM}?K_njkI`dnCHhU2%R0+QaEO@m'
'01lLx-F!xwg|Fv$p6H&w2_7`WEf2&jcdAoM_}}l`vXjz^&Du<CiI2Ec2!31Cu}Cl1K$#k7j%uTU;Fyx4-3=RSsBTC%jUho;lxz}rw-X?V7CR`A'
'Eao9WYCH!0vBa#{+`0~puDuGsmD`4S^mYlZ-gFp7jRNPl#vp8hWy<X4{F88QR{K&a)(o{IT}`9(mHk=eRW;$;soG=I<3dQF*@^IWx6`aAS5BC1'
'!pmWBjDJiuINP$*?RXS;{)kwX#+4waxmGO`;l-p7VO^<<NzwVAf%s=@D;R}mXCwXI(eJmmBEs`DHuU`2zAaI}pKSm^3omyMtH>^|{OvqHd{je#'
'N?Aigs-5RXGcpMRHFYO~Wc2Rk7gDRpA6F?tsZmD;FJW(ycH>=26nJwv14`z3Wta>wo{{qW6E{75o7|_ED8=uEQS+t{mk9D&W|mfyD*>MGIw2DO'
'@hjog4=^Dg$3F++W>qQ%bcx>L8t=1#y;AaVO4`NVF>!uv8c~TWy%;K19PVG4JkgaOHz`UrrvyGeJ3ILtzde2R>G+c)nQ%XBRV2-Ggx!jS0Vh#j'
'OV~4O+)6BxWj3Sr+i`ud-4g|ko)+r`Q)JABByC7Au^S;Edx!bQ`R??+80*>V51&qOr2XCitu3U?K_bL*n%+49UFgc4LMl+62{cr+!!(SCYU=UQ'
'MNB+RjU{RrL3MY{gg`m(Env^O(j|G*!G-3a!s1H7w${LH(sr8YIFry5e9wM}%+eIu43Hak$aZ+dmFk{7t4mfB1fy|eXzdyoGgQS&T%ts#Yl=GC'
')q(5kyv|Yqr}OMGWBaV}gfENi(8+XMV7W{7)KW$qYl=G?cs{tG41}o6w@Fc-q@X@scUybmd^~)1?wJ8^m}iwCfGtpM<d#F)&$>w)W<XDbL}aB$'
'?T%aeZR@;43UGeIV=~?fOKyWm9yH|~n2CvepnW>*A5J&1O0Cceo`SfR%+gqzcNp++ucmz4qUnTVO@kqDtG8GkwG=nBFb^4Gi}W5X2ENt%$Lg}x'
'V7b){LBlyAR;9}O<;0p*tNwW>((~x=n@put8Qa9jd4~t9HWL51%zkV%?XO>Lw@9p`>kYizmg#5E!vwwlefJX0lS)U9R@0pi-a~&peJd}XNj>B*'
'N>MB5eGmO%+QvMg>@1?XY|B%`C3+y*X0v;rmG>5{)~d)a(wHty+gn%2+^f9A^)k)D()aKZbPbJXt=Fo{BK-#^e?YqM;32xdDKKA&VLFUH(Jh<a'
'8yDo>@UL|M-seF4$govK@L}VJEns{VFxLmVh=nZjH|#x|ZCos?o5JbKe<?r+B5PT;y2Mu3Cx5hLI<-xa$C|WT1`BnH31S6sF(LM{z>D?)&LK&T'
'+!+o@6Hwfeu{=al0yDCK)SM2lLp4THF#|y(5qrU|Zkha=S_y26&)_4cusoV~BJ<A2EoB>+u?A`6W&RXm!6yMwpaAEMj(g<1ph!q}BV37*T#J#l'
'A|w7|L=AG9u%ZcxuQFOnQ&IUo3i4Gl$trh@;%US<K5`HhrJSU!0m2U*-sQs@we#qgQ$mX7N`+KMsV7eqKpsw6x6U-t$y1(9MoF{6#-lsRt7y9D'
'RU`s?1>W`MP+vi9$$efy;bodHFvlb{dk6pCVeWePiz-aM@*m308ZuJA1ODP|&<wKLgfgCLB5YW#k_;#2sba)Fc54~2$2=9CkT~v4S+UAI)tyMv'
'ZhiZ;)4=0(Sv9^9vY64pAjknmGBlfM9iqRBt?J^95h?WdI<P=TF=Ag^1%x7o;dtnBtR>Da-sjxn?BXB;>^%tveQi5(1yq#6RW2x9!_6X}wyB9I'
'c!n`YB*y^`x2{sRWPyQ?U+7ZM2YOgx3e?gmHBg15H&6ILoo|K@qH&U+hnN|VzxE!uCHfIMGm(~)WF=e=e4fgt!+|nQ34izj7)F7AI{M)QKYK8r'
'ZTQ*7eD(vWMT5}PDawe=<T;^%5yX5L@CEtd3r>lQgrk%Xj~s}@F#IdDCPS5g$go0#Lev$DU`)qnX8|=4V4%MWMF_Gaoo6e~B8x6=oQZ>*#5i(G'
'*KLV&!1PFerV=)Cvm^3zx&;7w_)nYaJB*3MKRBr`FJm&1l!@&4IdGS&`_&buTQ*ZW%jW6tc&O&$9Hp8ubu%7P2QvDubk7NYcrRc#mmMp!qZI{y'
'Bd>-f*(BChMk?akO0v=D!g@HmahLy`>(x+s-zcvGPJXfkp}06m;8?mOMWmIzGSN%7ww6JM#j+e4Sj8K(pC6$*_c{FY(quVi0Cp(#2Y7$@0{;2E'
'RY=6!md*SDOFXwq81dRgGtd|QYn76UVhPUg5uQU?WMeIg8V`w0Mr=d%Vl3f0_imQtvTRRqs~j)y>$`v=Y|K%FPec|IIj@R!)%OLqY0f!|w|b>U'
'9*n?cQqT)b#dc7GG6N!!EFV+@ND%NDRBk`qZK8+!O(#6p*Rar(2_0~88S7q#vNtbQ6C*Um4?jT3_TaD|4PNv+k~2_N3%ga)ddtB14Xq)Fm<mw9'
'7(5OB(1$M!aMX_uU+&oj#pB<zlkVd?S>^R12<6)tXQZ6my?l<dN8j(|_X>F0;AKDJ62MdVH8K}b7^V{5!)$g`ZU=$2K<ClZPEZr<5wl5#7H5$t'
'LP5-$P*l2`8M{ZurXL33pSX!P^=S1nPk}CUED{$OWn=~eUFfeer-4z5lN;zV|G~))OuA4d#{1WpA3|(p+O!kOen;`VU5L#cyGX9-)<Ycac1M+$'
'p{hh73{iK+k7Nm%icyF!;;H}MMAS$z6ql$(9()3|rdbtp#S{37HEQYNY3U3#zyeWtx>Mhc(MnS_3If!8Jps9eQoJcYR?fZ4$tE!dxts>QBBDvT'
'hqwt-QOs>V9m&c)9p@N!nDda%7`kg*7!_L1=#(R(O4S0s6^Ca!xCkY(Aqu0`t5z7#G2&zk8>ZhIm{p8{M`DZ!$dB2otz*QAv3pW!aI0U}25m%d'
'7XhH_v*LBJz~$8j2=ag?0DCH0^G~Nbvf)jRMAJyuZC(hxDaZK|M@~I;^pM@cm9@(D{DVS7B)s!oN7E&sjfw@yG<$e}N`n-!32eQY$hn47F4E-i'
'<ekHq#mjMba6YCvIuE`EMEL!46yMPGO%$NJfZxl9O6t5D^F*8z6qI{SR-hV3hc0H&EXZY)t&jWX3aH2fQxxpgqpO({1Gg``#8U|ZPKaT1CFWcg'
'{};fk19*%LGCw{ela1hXIm^!x)A%>0FB%3nN4<XkDLw_-c>gIazaNrAf$`*T@Z=AN^aKwdXUGzU!6CPl%+u*B04ptym(#PW1Z09?aG$c7o5^g('
'-yJ;-sBTF>d_%^N!{9you?F7&<V*DpmHAQdvB)qQ<?DP7+b9ao=x}O0_^_^UPUX*6$>dsn&oOJxDV$NWNfdkw5@3Nvp%K_%*D2J5uL%41;_u(!'
'FRHXKSHUqRXTvsdRA<ZpMPE>gh6shE&=&>olHxL3zLj4-$uHkU!HMvbw`M|2px~v^oAe*ackAE`1ls86W%R%Q-0y!wANT8k{s&+G(C1Gy06_}>'
'H|X<!^gVh8{~bmk;h>Q7d-%`S`=9gt+Uokn(a+HJ5&ZX3c74>)^#_b5Z1w&8=jvMx`(;DlwjA@z0P$P`akw1_mh9|BctO2!*Y4MSVR-%s7=++x'
'PXr_3(b4g-=-K%CO8`NVsSU!5Eo<?NKJfMR>j8Zt350Jg;4u4#-5`MQYcE9qXF!JMCPqFd09>MYLMe7Ef=KryNU2ENI6?itAVN$kw~?FKjPeP6'
'#yH)gAA`5M$rY4N#Afp-!+gz!J;53bjyDV6h<Mk-pdURL+D8(2Hs&K+UP{phV-v}3C}vNT7y287u%on~nDHi?fH1G}76><8CE@qv8@jkxkLz)E'
'b&w6wBwovq-UUxQbgfI45o@+Ia68`U32iFkk*<}&k3A9MK2PQu@6;u2tf=Mc{`2M*Xi_`1rBn%0#PvBArJO-^V{W;|&Ml<9>UMDx2L`*403pN&'
'wIeh<l{iq(rcl!fbv6&SQdJJga(mFSn1u_32NcyGaYiM}qDP!uQO7JX@nhIHA?6f3kNU$t{vV{OX7nHQevjx+JXqoWgQc0+xMQ@z)f;2`_;dy^'
'9w0~Z_m7Y>a|s}U0t|ZnZHTN&c7wy-QD@KT*^VHxO?b*uf;&4z#~Y_dk5<o*p&95=9B?j=ntGP2Wdu4@^pN_#7Tm~Cc4TEtHZ*R4CkzE)tA2dJ'
'aA3-7!F;yI&2dBe1pgHrfBWY2gB9gR$1kI3;W5Xt!B&+1n$CMQ3sIecp(#o>gZui!o0HdMr@pT8dX!d$h`K$}ZeAU}IEpdt0{X*-SLw|}|C(nP'
'cktR$c}ji0Y$XcX<8!-R^skdu1vL-5oD_ww`R7%U%o0v++Qm{`cKTo?>ufX_UT~1VQGZxWx^vi13A&rWPqbUI9xN`ZB27nh@aa}-&QAJcTI8sA'
'UQX_n@~|<=?YHpX`|nQPzvjHAbF?lnk>MdEk$@Zu@h%sRP3C;eW{XvdR<h;Qy2|FgTfp|KcF2!Io6_^cy=>rA;G}YWu_|)Ru+b~l%l4Stz|mfz'
'diNq(E;%4)S6~#Sye_LLdcPQzKc0R(q33B)n9tvweE0s_w{Nw_492mfv`W^+kNJU-T$p$xd}9FK9{0OFc9@NF7HX`HXe5(=;-AB3p_2sEz2V&_'
'n3%%E=>IAfnqhuscVw2JqnW{06$(!`jv3M}@oFd)v(hYkrwR7iH{WkFz!p|~b=zJsbK1sJSeUaZrm*9++8_gSPXxPKOPud%6pYEb(~M?DP3TZ&'
'*s^0;Nc_@Iuw8Pi9O%qKDzuh-sBe7;tUBEEc-*&dGI=m+HI?Dy3^1U?i!5B=wi&TW`3yupOZ0|yrg)4t$^Yil$;sJ=Z{NT1yGUCrLe;me#Ey_e'
'W=+4VpuP8$<Di;`US3?H8{y5N^c_T=e&B{-+vZ33ndabu#<Hp9kgj<O3KuIt7rsdi)fL6KAAJ50JUiz=o#}jvW-4JejYypZMBye&Z_7wnmqq9{'
'M{m$w4apL^tD(LNYCU(sa6Ju;*PZ$-Zgn^mE>cageJxi*i?Of^3CD7>p{uPs)u_n$L)1_Rjvv=iQO=DOQ-hy%um0Ixte(04dE08tMU5Kpb>BuW'
';D(%Nc)Y-V3*ql9)LYPXP3&Q}x*f+5nmiGWw%N$z;=Tgn|GZC@Hit*zk0-&27NRlrjPUj5YlSJ>7JnlF`wGXE(2<@{O$SmWTDZOdvT%juy={y@'
'l(l1kbiz(;cN;^H<aU=yRDBEz)&M!&dL_L3*ov0MW|=_^UK(alwT076?|z9{uV4?wI-}4}a=>aPrpBEM7xdh`X!k3BMm<hyFq?TC)1+#`$c!Y^'
'9CW_Xk8O%<CQ^dQZRqwlqT7ZtNJ0&WED**jSs+dosb)t07NmJA9T9Kx`FfFhO7aVXOWLDoo|sj(BBNqkBBrMjQDoYmwy#@sA88b7K7T$la{~Ff'
'A44E?DQ<e8T22w2WRZ>iRK~ltYzY^`2SWgI1m}~!$bo_DY~}U+pl+lo8KiwP*odX>y&rp#={%d3lZ{>)^xHATFfZAhg{p$-_tLvfZVXE44ISV='
'nb1_B0Nmx=1`*FZJrM~xQIex8=%Q`F)Hz3sEqFh&(NxBIl}G%?9BcTw<jr{1u;oxgRDjMeyjT;l6Z;Z{O#OrzFxY%<ySX6!a3hD(?A85T^w9~%'
'(3_^_n4=)YGUX@~){9|h+{Kur=gtrWTB)l`i<{|b720KYlfxRRM^>EZFSV)FTS^lCQb(}gZbK_ZV(^ZObvB>I$$FaQG2zL72gb4d;$58H2;&lB'
'7kC<^Cys*OP2%cfNcgF%f`#LpK4+Ru**>1RO4Tb~U>)%?VnEp!vnH@nCnk@wT}hQB=0Gc9YdAIO9(lp^QtbhO$aX1`#rj?malk_X4Rl8^QkH>a'
'C{iW9c*mn(w*_@=K?f~#IMhQ5o(3(XuC0zawNcP}_S_yFsVyA}wX}BkicpOkK{VrUYA>TeY<rZ}Jj&`@R%O<rOu`iYa~eEoKsR3r%q_1oG&@nO'
'd|hUfM&0DQZ_iF&#~+VB|Jip~rMj`qcR#XR;hp8gfWOWw?iuKG3-(3Jv)~yzfKWnbr)*6d%I_&mi%6!f;TqPqmUJc!sa#n153RKdW2Ym?c2_Rv'
'SmJ5L<L>804U+8a`U*Q3=%0hizoSW&K<^~*WJ1adfs;(?B>LIWOm*ArJ0msfKW4xL#r`xypf@0nrXH=WU_P3Vs7o4@sTS?hFpcY$wjmwe5G&qc'
'WpM2@PLz*2p$8ZdBREvjKwA2;h^Dh?HbrSUo1!b_Z@=xB7l{-+4rRC5KR0Mbd&$yCiuvC$m@{c)<MeJ-wB;|!ggFN)X?KL;4RgO=^ThdmY{D>n'
'g|Y=Effgp{-+V*reCz?lNL&<3Bb%WS{EF1MaC{3U3Ci5une8BuEh{5>tX`6K+leMY$9t4d0VWgDjw}Gg6#W>fpcq$2B&Ct|e`*R$rISDUyh@MY'
'9Fz#)`cW?IhwA=aP6P(Dd*BLW{5Cir_MXq=Z7%t6q%&ApiVVYd0u;Yz+NQ%XT9R_@NOU_MfEa7#L#gmQif%1^N+dI@j8_BkYJdco!@9_K<%+yx'
'#0>YLWHrE9?OS_7uAm%C`+>WnQ9KpDu%7X_^Q2vMA1d+V-~oH$@1aRPtvSkO?RnL%qK9h0MWY9+^G)=iYp8~l%2~d#QXV0?#Wz{G29C!1%{D(I'
'yLQt{NqU@$AF~zvDb+FG_KFdBuMiu%K-m%q5~`!nD@kO(OBLEN+V&*!|CnP&E*JvNA8nKpt@c=NZ~E7y$>d6i`RwWx++LxRKW0RR&V+s%dK6`*'
'uuDXu71$y`sC$^ab#g7ztYcd0BAs)pX_19mq<Tk>s-7CL2`ITpH&7Oygr3@65z<Q-NlO`WZ~<Wrq5~#Z7%_R)%9jTZFz=gg%Xmj9L9gMPED=db'
'<nEF1-2H;E+&>bQ`(F?iu`58<fX5?#M{6;9Ae7$>AB6Ah5HJhaBnW+%aPGx0MXPjD#l+D@*>Zw@&1v0mWm4p;T3eAo<<XC$)&qXuw1#)1Ry(`x'
'X153U0O&&R`BU_p>2zAdTC4Y1_^O*-S%vSZ@N_g7ma@w`4Za+oG-VKbpOQXvo<kBX?fXG}bGXH)B`e9fAFn*vm&tfp-e#P4k-UY-xy*b`=Ky-W'
'0H%FqzGwdvaXQoiw$O|>HnKAc9$3oop4#ohpT%4bqv?Dv80{7}lP$rcM2X$*FNU^2D0>q)O)Ldf6|K~@2#;`Rp;5pt0O~b0WF>CQ@BZRXP_~c#'
'f#CKv`vYG>5z(NLPDHk%($|kX;wKkzL|MU+pGW)q!gQKkaP&}ACx1l8oxzUr7)ZhiD^X*TWJ|O-lmSC<B&ykMMqB1tkPeo7jYdJ`+%u;Zd#$8~'
'hTNc99}#l}9vG`<uFx!9%S9GU?m{&!&~qgIc1&HfrSj$X@q@%?Qu!`cM><~I?&44=lxln;OcM~E2`yw~=f)G(mSYmPCC^E01t!{8HXXh`-I5sZ'
'L>*q_?5SzBSHV$?Xg4`A%i8lv+XZqf)QM`fqa3yK#Ufcwoj`NiW{2V$*DQDD`Q;pBWx8u7eJ?#_AuMLG>n^UYFwq*BryPdOGc4k<52Wo3r>&q<'
'70;PS(dcd~DnQo&=nki4`t8nelogdXu?n@cObVVWH{6n#9CVEvx4N@d)S4$Zvkd-r;h$M+lF##EG#T~p2K~Xs;RM;Fie<tV2$E|$k0lc=bOmIk'
'z-u4sgs1BxJHy_K*`{S&gnM*mCj4QyEXkAD?Lf5lY_?%zAa9{&@~3NN;-ZA#cFX{N###X^1B311Q_TNsDW)wR$j<SuPMYHYWio<AUFF5G{P1PZ'
'8+gi?n>J1k<wlR^dA_2-46GqhT{0T9Z8bZ}T2NN>)iZjv#;o)mb2&~mO&wP`Qky+gyVkj}hwPiz5GW`&UrscrJG_*Kbu6fTm%$-**iFt}jZq}p'
'Q#c{H36E=4Yd6`f^L*40YK=AwCV1+bHbHstx9(<4S_@NsVZni6u2E6(?3{z($;MjeJ6H;xFU`K{W=*%rjiGMivP9t>XF=nRRp?Psxvo@SWP#oB'
'T)K*mT`PTx6^DIus5pheaAqnlS`;sis(Y>pxmiP+x{;y&mVEANcMjAedP30D?=_R^sngciikCeF$!xzX%GRGm+U{wfs5@zmQ<%dHR!>G#3d(t2'
'eFd^8>c5tJ6emYUH;$UIoRfr<&Sh*aC<&O}Hx3qvGVG?H6#R`-Y6QzYn6(U45O?64*0!1=h^{YeuKA%e8w~qG?g|XZqMpT*tM&3aoid8`BYS-!'
'bs?85FW1Rs8tLwvLowERd_<T<-`awtFz1x!7pd8}$Sf$*49fNKg<>?l+4qIcSPn_d@ytWgSi~kP_;uW(@2&Hvustyv8$X0^^vH3qF&cHc{cWJH'
'sY#VoC5u%VBy$wB?lCYQ23)*^-9@jXV7;7Y*Gjo`eD<m52nYffg-lO4Y*6p@3WJ@E9$MDHsWmjCE3{2vVAPMaz%Wwk`E%Fx{Bfc*TDPruLhNQP'
'^69;zi>iz(3)`MsJ#oLrF%B76qU$;$?L%v^u2PWlgwCh6v0c36=aGWFIoXzgFLdvEG*1l_u)@#Z4sR7Msr2ZADb>ZOhWmZLK&9D-<Q^m+J6zB5'
'SK1*!ott@k#4U`RNaVv~S5B%kd|OQNgZOy4iEOO!7%DP|Bqd{jm5<t%ooD1-g2KNkBr9`WO9l>;<$69Zi^*tmg-512hQKB(AUGYR`LImze1wX='
'baeEx-|w4V10~+b5`*T0jWLK_BXcA)w<T%cNDEFUwTAHGRvbuD$Q!bqd8~=tr?QH1q~Ruse(=F=o?HN+{+_8Aul=@S;&y?Q`DM%+Ji3iHPJvlu'
'9t5@cqlkiOZ5BK_^oa%oRUr%ynn<JmzHMyn%#9ICRS>USy|hE>t*aB1dAD4tM4lycV@>#{Oe2p6fqY=h!Y@!}Pg9}HUM16d&32AT>dwI-d_xU1'
'e&d7n1rXosNb8vmxtH-*U+j+`vd!@3{K0~E6JhL^q!<nG%c?3zA3vXU&psc2`V3#jH?X`=?cB6|tRbq=g|Gah)Co1H6`?4Y5DX`CK_sH2Dzb&*'
'preN=o%f@}H;}{w7@IEaB67`hnViEBi76Gs7A2f$%pq!}1Xx(*D#ZbB<I5xi>Ue~Z!{Yx8qpD%32Yie2(NOB$K6^~JHM(T<=9bsjA3oA#T#n!l'
'@%Wn?^~XaQOw-bT0bv>_rNSgx#&>v05?CqF?FwI;yPVP_z-t$j4kB-@BfDNQpSKNb043m2P@X;&sonojNj`|0XX#q^+>=&i+n@k!rrzAxqA>8<'
'uC+WwjD?eB&9zizIbfWePouNnk~_qYsl4L7%<+bq3^33p5oST&IA==`8}JA6fmR8~S(M_+xa^q?e#?S6O7e5SC9GoH0Au3CtE>#j`LDRaWVtt('
'<eQ5e2pQjOPn`yom?<2ug_AUz@k}B*;KKld1SRkvQ)#;zR7o~B;FIf0>IQT&=QXc-hQSR#k_$A>&OHiC^D*C*bn$M*f0p9i?6mrH(9o{&LbvQW'
')p4sN5IB-R4FT<pBZ%r4v7%44Nf1`k;*wAxZByFi?g>FiB8+=h?=_Td$a|Ruf25tz*tZRXLi1T`d;^b=GI<mP1e-%RTMtxsQz4fwt9%vXNpLio'
'qdJj(8w6BB0}D*F^ug@D7K7e~QTAw7d1O`rs~ukhB8HWk4z6L51@0qRm8lLrr|_%qN4vg*be`2_ZJb7~?MR2gP(6bb^0c{_R8y#Yd`ntiPHPV2'
'3XQ3S^9|9sQ4?ykG@PS{8b#N?nXYe!Ud+Ibx&y|KVPj7m<ZkpJmV&X#9KRQv-1Rjp_)v+L`DtyPc{cnc#(L#jZ~S=Ky)9Pb$L3cS^pBBlWDL<d'
'^?>vA3<IK1Kth9Mu$U)E#-qV78TH3ESn=G=qG$sb%2JKuwgrXBN_Rrh2Pg*Pqy~k}qC;82d#?S(r{otO?aujboDL-%?WN8#R9S)6Wi?ObEPOC}'
'-Ux}lKgK4eJJzgalxit8gK<3}?zrRy80=Q(V#2x_bLb+!Gj12OorWC)hcEmvA`l3#%7B%h3M(&*%R3E|eMaq$q3*FFYl?h*W;7_LE?X2D2WJH8'
'ki+BCNjP{SnxbEw4o0QdMMV2gtnRST{^#wVsly$|^ndS%TM5}0%tQBZ`-$H`B|7_4!<$GPQ;W-sBq9LCfpwAHrNt;oMqE)Q8{wm%b3nAwf{XYL'
'i}{V@$mY(dFk^QWqZq*7?n#m74)m_$o$`C5rCN9-6O(nZHQ*<uUa<CpjPJ#DYDrj(`gde183p(B7qwwk(Uzm=MCQ+X&;HpXQaFB8>xRWKK~D4A'
'i5_GZo2OS#+Ji0{GHVlq&cTD;^i4ymwPaa6&$j~X#vOkg$ld=NAiJ4S^;+%*`EK*{_U8fl?lB<Ovami5=I*~1W|9`SAccXlgTp5!Y`?=NA5WjS'
'@Ew|+dAiyawcj9s=*WOw>tD(;8EB$GkMM#*{|3%*Ta)|hG}SIoM}%sWE#deF!VK~f7-7Q<M-DIL86PI|)fGxXeoDCof^$u8pBj^S@c9*#L3PJ;'
'zXl%+X=^+f!h~(k-4<94b3eKpwH_E3H?844{J57t(tH*t6Jib+uxZHl0yHgr)jw)oq}6STdbP*^H+pE2PZ%}eU<Z$UBe8`<R6m;qrGzD8Lz=2a'
'ltVr7E4cyyzMfCxi*-R)lu#W0wY}L4H6jL}Yg^ZxENGGjDlN<SA}Ms=R@z4p{hDgai%|9=c5AhFTT*ymQg3W4=>em0LS(Q0aEEQG>_kaajk@6h'
'J2fSyGu@_H7R;V>#s&5YKRZsgpzy%V0W{>iKtuCci=VZ&?f`QGY@i!JYuWb(&`-?e_5+G4B=HV}DRBxyt(S2W5DJ#*dU730N82rCP|5PXZCX)@'
'TVkKl6`czNkbFO00+=A&Wvr(}B?&&pr=<20+BeybK51AjMuW~yQ!euy6F1#Lt5(yy&bI#Cla1QoP|VohvcNQjeIIzzY9$qC)atw8S>;8Pa3kG)'
'Ut&P+yZDheJN#E%owzqDmZ+Qv&ViBjfJQQVj%1l`@1<Z4(^0?o(t8afbdit0`}vZw&=kWM@XzMI#NX{h^uR}U{tD}5JU0iSb__u3U=zxbH9BOB'
'hRr4s!$fhR3UrvdsWWscxAp>-x8SaswloC_TzTDKLacVFkIUJ#WZXYz$UW^i3PjZ^SEzO36jx?dnr3*^uFfb`*{7`*>ww%gI!~L_nhAhDVw)M('
'4ArX+Hp3;JRp;4{DT-GT(=fvskjrs0pHro#mG(?St;()d6uM#6%tn1f(=251BQ)a&Ha4G*di}%aLsl`o941u-$Nte_e~8IzVGoZEpZCXzR88w_'
'V5V8jfKg+DPq#1b1a+a>t>6wUb!fpnUr%uXMyI3h;Ar^yBmC(-AHMx`JQ@tf2@tPy9SC3_6{HCnss?K~U=%QP=5$FooEG^Cx-^vSqki;-Ad9Qv'
'9F{WTKk;nN>BVZ%eAV1iIidg}0I5j>_VJIw4~HwYf@`#l;2Pg!Ak~kbPR>p~x1tu_X<|TAkEyL_M;YX);HOZnT(8nXd{<3GE&Kx%mcPIsG=5kw'
'u>3r~ZMnxO{P=83PnS2aA+6w9CeZFRlQp;FmI&9{iuiBT>Ix?q0OQDX)-WrwcgdufwIY-Rm5^J^<dvjji)OjmprH7bKtO?)m=vhHNYD<?kmU5k'
'P+$s0t{vWlkwldz23YuV1^7B!P{7&y_07>?pH2G;VYP>5M%+e1W^W0_VD4VfPuP4a(hhCHi>yLvXmd@45^@=0^S7-Qb&LFl;K3hHacN}+DTIKp'
'edIL77}ZP66-7jB|Di8LCZ*whf)OuacfJ6*341&~4GiB;>6R8q|FMnZkjNccA+^M#?a{P+etyD(9nMlf4$+u77j5bIQX+7fnMGX;S(d0d&kDf9'
'p&W~yk<rOfELyW1Y%vxFg_6YdRldQQrIiojFkrEifgbA@=TY#?>OdwHmL>W3xmiTz8J1<l1FH}rt5Jq=ou66-g^%wnO)!x?4$H~z9M+D}P^R|G'
'v0DiJHg!<b|H7><Ox4(b$*hi?!nzD`(?N&6OGl{adt3*4Rjar6IenKCpFR{n{`k%(cNbh`ONVNv8sHp!#$$M=!_KG+Ik#nY6|$OA4N)j8Y$A@t'
'|9!U5wI;PcQDe*cB=808vyW3#)5g*HQO!PiVlBP#Rt65<7~cKs6@`0Zx?tohP)5ITCZpMy9_Y>P;NGy#MCxK7C}?$>l>FO}ztDZ);xN8S=NRl#'
'R6<28JOt{WT&9H?5&8neO_EBG<jF8K!Vx)qb^Q7-uihU2=SlqP+t+_N`5eDLes{vLzmU9#p5VXSo<IJt^eCpWiICWAhbdMrrl({r3=_6Sy={_p'
'g(t-DK~LjmSmQ!8c=WnPI}Fx9bK9>DQ9}9V^!v?$h(2xjNaIJ$3e>Tu$@IVyk9uN_DT+C3J9xm6?i=GTBDp<SPz3yeV=UP))gQlo_4f2N1p|L|'
'e1?;3#9*txq-d!H$6rHbSUpDRLr8-~+J}Fk3XDtvi6OaNfH5i=h3bAlI5!6b+JW3ta`MPsnFd014b_2SS$9VsgM%whqVPCuhp!F;o=_QwZW#Oy'
'r5TJj<{6Nfe4UL|$0mE$@1Kjkn4mphKmjb)ODlTCmOzvL$S-QqD+2cM83wQ6?1<EO-0SuDotwe*OMK+Hpo-QGu=11Iz?0h~GYe>+RDfDSDdr1U'
'#dU0k`1eQh`|F2Ke~I6menL!Qf&xvIfJVUqO)C5vhwu+<HI;NjH7y+H&`}hKBKghXlGWYlJV7-KLqHxDLyxIxF4IxSRvG}6)S_cpV;QDfSb>g2'
'&16EJYa&dgo)iVAj7x~))3p#gvb<O2!U?}^NRW}}Lv|>yn6sshyNS>A#g5K0h4oF*F6ffIgo;$hxux(Z?3UC{At{E=<ff6n)=C<|EQ~x;J7npz'
'bPn<Xh!SG{Zh`UTx4~6n2F-BeW%w3Hdj8Sow%2HID5p+aLVCt&`NN6uwT4q%s)Cbl3F=~Ckj)>Zvs=?T*PS+H=aG@94P9IKN6mM$hFM<{k^Qqd'
'iNSiQ!c2c997+zVjLf8GKy)rJFxI+s<1ii>aTt9ICB$c7S@ZicEA6E^e92)mSf&}FL~Y#+u|@Wlc}(w6daq&`GfK{(HF;{PqTuMIm|xieU^Yhz'
'1L>>g;6o3_A>KiT=j7arCnq*dwBB`O;$S>Oas8hb#U({1h`tfx>PeH*gCLYIDThPosbHqM(xbuBz3F<fDxDM?&M6B&Oo%blF~R03Ahrr`U_7d+'
'RLXX$Wcb%jY(e$)vWugX_h6xl;%SimEIkE^l=L{uibAysWI*YGvBFcsSd1d4;NinS26QFATAZgg8CHih%PwH<m{3Uy7<}kFPM0MdW#gorWEn>Y'
'H?w#c-`%uhc2GrYW9dp_i`55XG;Hj)XGf8P+s3}D$9t?bJKAf8w$K9L)O<R$slXA&uJnT=8smzTQ2i~cM&5|uau3NpN4|Xc?-Jy9N$<9h57};y'
'YnmHTfK)o-gA7ROb|Q-XHh4Wx7Ap==-&HCjptF}i$3zT4=z1=FM1*4p$VBNpU!}siV}gyOQ{~DnxR2q^-ITez9zV<04wILU;XG;Phcw@*wCnXW'
'>UA?q(N%1k!3ipWLIx9Kw)?5|VV)LbJ6b+TrH0V%;;E-KN`PsN)oku-8%TFZ&U{N2+2}uSHD<{TMyJTO6WAIf0+YW(S&Wd&kLYX1Kwf+0&;s27'
'%u;f0U2734Kyq#;^p}U_%(AN58|8|0kz`9w7iLPN!p=&Gq^S9DQLgJjwCyO|={z3MHH3a+;R{%g>usp#fbBtDKtt6_Gv@h*`?U`uX93L2^1fgX'
'p@t2auI_Q;Ca~92(H-pqZ3`|s@KNr{hG>1l&BF=W_cI^@H&0hTF(n32L7MT0I6-fvjhs$ECLHq+M?n?)nj%V4=!|M7SrnNId^v#C4|k@-K`k=6'
'HkXr|)JW=7kJJ!seALyt7F1q5G-r1=5(=TuXjO@Rs@IK0<Lni{5JX0Y<^mhZo*S3blXUs#Ur3#?_-9;7+32Ge)2!HaVB4~wa;ElPQmbXPs7CMh'
'vKp&pcc^%l5+6y8t;^T*kGKtv&)r7K^=dUwktig^y@<BpZ3-<)OmbmQO)X{98MK=hln9d>Rk;Z{B@@un{037v-IFMnQqFaZ53QtS38ASA_Tznh'
'R5}4u)1=smx$4t0yKAhf6T<9Jd`Up7BEyi|wy^&EC=weRvm<6{N#eR^fD$=me9=X+%w{BfaJfCib9r>;rK`o>bX6d(jBfX{<y^vQWmMx@Lw)fw'
'lBaixJ|lJMr^aXJ-D*}4$&jZAdui%->s934gsdVGAMRHPNu<0U$5{Dz^k@%iF<>}k=TqCtQFgdJfUGNBnDLWMjB_3dL~u5s=S(j`(Ap1uiU=n-'
'tv^8~LLz36+<Vvqt7$Z~8Hn~7t-=z%57x(6fhvB;3^85+*vqfo+p4+}wXV*XQ7mFgrigH}>@Tt_67ge(WMYQ8vKj`VufYcsDt+LOIc@01sne;W'
'5TQ9@Xj_I3J}1uLOmCAiq28d_1hpm~LTm^1H(rRXRyXx#d4<p%pT0PZgD2~2%X-au3FP+B7ZuQLe_StvANDlx@nPk<q2jB1d{S7hC^29}n_YK8'
'5$x0{=!SMvOI;7rS}$00$2TWu6g>MEjGsJwnq1kI`7#ZE(P0~Y8%^BGB?)f8dWU8RRrPKvw>N>cr+g1Ws}<wkhTPJEkv8<|qnI6c0yQDtZx$uF'
'=GsorA2|mdFU;69nmTvYwRO6qr0l&W+qjl<wt{SMw3~BS)=eV`xJ<!p+degBdzPd9C&;E*buT8AuP4kY{)vQjEFk)lhSkV~l;s$sY+W{e<EsrU'
'=(&W<bO*9iQ_3dI<MLyKJ<=*39tKYoX*-~kUW4q^kte=#B>w6fQCYr7XW5cYDKQ+kWU!vEEt5m{Dqh#o$<y(PNlHwk89R!<ma+v<QD6X3lez0n'
'`=z^)e_y~tjqu*zekv$9|M612Q8V!y`F0r*uaS?0v66n|z3EZpZPY~QXyZ1Dsv|EUl`}k2Q*FO@kfMQzjy%-VKwS4I<5@#xKmOa`lSY|EavxkJ'
'H+WDb=d-B@Nu~Ig!0jGXqVU|}Mq-j?t4g+~bB7~tVu9OiNeV=g>cpi*3BoD7D=?ASQluZJ09h4=5SSLpOf^W=ggICWtd~P{&65%_5YE}i$LVsl'
'=Q%4iABg_{Q_ugvQ_ugD1A=g4k+dD<Txlt3PlM#dl-*6mI4Z{y9lOEcoSD4Cv+M2@cpS8v^9a^%cb@Yge)f~Etp2yjTIPpsxjduXE`yn$z2!0U'
'<Htht(=4Z%pZ#SvyLVfDYhLDUvYh$x98EgA(OrK03*Zq8Erjv!``sd-(}*Ct6w~Gx$TlSHRJns)w9Em+&=c;WP*yjVl&ca;_;;>#5h2;3ym;`|'
'5NApIF;XYJB9czkM!knkiu5NbwL^&<iCM}wQq`mVSR0iy_LGDF04sadk2W9i$x`<GdjzU&;t;EEw~1R{XrUsjG7WIBXOFt}8jkFr<^YN+n<W$U'
'Dzvv<XjyAX+hOxr-Dkn2ROUj{@0#NTU#m75a!CWGj<vpCdo$y75Q}oA>6uy8v^+EH&19{nxz*-V9*Zf=YVKG^ZPDe+WTNKjZh+Tw=o_^ztpcv7'
'&N#diSaZN5b_LWAI+4m}I*pTR6fg7Jc1K)ROtXS2=$i`|$^Q#bO9KQH000080001DU1Alut?do~01h|+01yBG0AXogV{dJ6Z*yf|X>u-bdBq$3'
'Z`;Q8cLV+(t{etQEyY+$++{+UH4ECIXjgP?&|eILIguxcHAS-gu&rhJ-}l~q@$N`Ua@Jr*Ym>)!@9y6B$Hnu!Us+YBMa~kRR4h+e#j1JE+NPoz'
'VM*GMvMRO}-IGmGk=NgmH^sgzY8G5xTwc80F_N(@jSuAAJMkfH0NGC&&ELHPe0!SaHE95BUnDG>Z)lw6TM`#J;E$Vtkhi<EzPy01ZN^}HU#1zW'
'2`roDO+jc**khTcv9XUP-Lovs8L11s<C-J|$%}^U=sml<fSKEt)iv1_4*=FyG2_#G3yDuMdVsk;iRqqI6!wUl_hHq9&^Ad65(AdYi;88m0SpBJ'
'I>FZh4padFJkPtl*i^-yMA4>g+KNRHNr4Jg15f})5nqjz;!w?@#M4WN*ZILGKco$;oN18qkkGtIV+HhgTC;y5>3s5S$}(UQCuDR_voxViQR$g9'
'jbg!?s4ReD2ZiEiRu|blOTGbS7F9~KpZMqgv^w4a!}RvPlbrsU{SH)UeBv<J*O-4mGU7%qFXD^>*?-|Af0L!<R3Biry1XFpbC>4HN}c=06WsB9'
';^nleIKjMI5)$=6Oyb;#wjfO_vdIeC00gJUG+`UsW=*&Vu6(j(;T8VPShx(XW@4^D6x9u_8h5ssU;{GTp99tyU{Zw6K&T*3VXRBWdJy)(5PZQ8'
'_Cimq7_AE+NZnMuSzgUD7B?)3)`uubD|jv{pz@wRMj6Yu%`UvYQi#}nny?&>Ua?KRB3W8DH-Plk#TtS70-8gJ5Eib*a)#VTWM4Z1I>q<UqZ?qj'
'Cl>^KL;|qr0R*CWsHd>w&5X=n5w8;;PYMnGq#J@DLB}TYRcH+fj&w6Pk!V^o@+-|+_Dxm6c06PLzO5UwW@HPbgzbT$0)t~BQyVtdH@K=dV9Ug*'
'9hevfw_u?&_fRpgGC5z<)VDX>etpjl8<@rVI)~Qoxah|;>p{4xwUDZ{2%cZdd5>druH>BgRj|C~v+zOAIy`xnd(^k!J?{GFJgt(aRPj#IP^p1U'
'*-j(&X|848(u9{15&DvQZq7y&R04p8vVFOubz0l%`CO}K-L8KF{S|orLw-aj<f6ktM<cI3UL`A<Z<(Qx2GNH!(o)5<G(b9B3<B=RxF(({{=cSm'
'OEd0;ekobpmkR)5O_U~_grCZa_jJMVMhgwqll?)chcsz+!>c)vJC<&D`T$3zV{rbzBQU=V_73qD%v)Ybisk9vl3p(d9MT&XC4_O!s201fcyX@B'
'+VjvSs@oDHkWcQdl=%FkqUA1)-?Rgk(}`AgMRA9*N&)e4fe%^nz^bU(rB#BjzqbXbPfW!d7}X3Q`C~)*pB;<ua6kA0qR0QLfEJ(EG}$um_Kz}W'
'4SQ^Ca*rCWj=*I`W432`V;sCGO57fB29BGtjW`s*u&N`28ebPpQ|v_;Ct-S0r7?54f=L^i@V_<KAQrV{_YBcNj&G&dL&feH%>;HnNfGOSE==Q<'
'Qh+mPjJv?62U_J2f@@YO)^KEM#u>7zBC051af<2FnC<@mhE}c}?<D<?G2>f-57VUo-HN})82=n4lmS`-sZ0TB9`X<tPGfwyFVdJ5_d+V`&7)E!'
'dR&i^wvuvrwjGr-KQ1T9=$d6ubLh*fv#hAdltczNB$OS!!JX$Kn;UVO$4TZeB^GurNo$@Ha`Hc<rQi;2;+-@Ht<o=-<YTd5=S8m1@**;NXIr#j'
'(Yk2!gjOd=fvGX3w0$B#ivI=@{yYU+jxLf_xmy>BA;oX^EFtTIQIEhH!oOYzyhNa4Y)r4Cdr@yO<#~6tct+DXV*Pq4vtN7cILYi}QV<}9>`57%'
'E}uFhfc^4?L?mLgh{tGlILrgVCzvpe@}383ktn-Jluq#WF=(Y3QW!fc_arl`B6p!dSox9olwo=Tw!eO<9I#e0h0RvGk2!8kOO6mCQo7Eta<DFn'
'3=qAoT4p%I2qbiqqEnP1T%0kQM@-d$q*UEgctK|`B0$H4iOf%E;BscofY=%2k!=qr77)g0MUph;&d<lv*Z9E^Ogl5-T_Z5`l6*Fjr!ZLSyl|1?'
'2$DooWm$NsYt=7)*(Vt(srh|^!Vq`8O<QJ6>>J&GXI*|4&XB@c!5n~A!oI<ys$iHX8gql9>y!y25QwRjOe>S>v!O3vta@&@$0T1KE078=s-Qh='
'AtVO_*pfp|_Yky~Z65C+m*7r7jCn&io)6%V0Y3z0k+cw-qIwWP^!VQ7?2V^*`CdMJ7@elLADOmo+7GsUH@fy!QlAKIlJ8yWG|YO(TPa~Mtf+MV'
'D@`iS+b|T_0nm^X6%VcEVuR!>vJ3|JfJpi{6jN$5+$$Nq1GRzR<V*4enS$q7m^^w#7KCLG=owiCybd8)Tqy+CCosq<Al}nOuzczB0aYnUld=%D'
'M;sAw8YiBS&jHc(Qd$Hacmp#5dX6i<#FaT8$e+nlue*3@35i)X>JGy3hy1tSjl$X?{BMltqEL8aN5oFu4HG$uPE^!H1o$Xom;-mx{%H(~APNc6'
'3bE+3KLHmWq}$8VLPMUB<p8Q%<GHJ1g+h+yH^Y+3o((rBQHep+nYI)_SG}Hs3XhEd!Rj)o%{>%VQV+X?C)sL<+fwSfCi(620i{N{S8A-HMk7=d'
'wBQXd*%!lO%3c;HRR(6>A8%?|0i!hisy`J8z3%z0N;PQ2k$nYlZa{c*dv?I5K+g$lG}^YR3k(lH&cq0I61mEPXo@%;UP13#T&rPIq*<`0wGQL!'
'nPGlZ6pZ$SKBD=#)_Bb3z}y`a=A=V}6agegv&z;d%y2@51793{>zYy;svB}(%`JpO`x=I4#tyn62i7WOhJxH9Ac#-$;|onl2UaM$Ue?lk^<$Ma'
'!kdkocD?89{ot9l#Gt$>k^wDC*wz$Z3=mNo4n=azD-<0c(6>!=yKo|mzU|W`6-fWErZV|T45h|&awJY|cYio$rzQf!QvCuEQ3Wa}x8b<FfHN*2'
'H3Zdo;5zVtV$Us^c|ubvz6f4`$A7q)0EH&E@YolIdGMmYl(lMtmSp-6Y*{m%;BpgxGMUYS3Wh=GXIS`wT)o*%r;}jflgVfJ>u>mL1#q{0^srlb'
'n1N&B4}7Z`YN!OeLC-IVJm=X&K9!$l^Tp5-S_T5`eQy@G0oqAGtJx|RlJuDmk?<Jm&b<N7Nx7@g=BSdvY6AF?a%v4uy4%^!3Q`GY$Cl|-63yRi'
'5fTBTGgr~d5v0z>j!T&k{RwTGV$u^#g*cP|1H95}bCcVx81V!WW6$YCm2@VEQ^wV}3k=t>B7##6e3cj+7}j@VsgjepZ6>ob6^0Ay#BeMV)D8z0'
'Um<)+q79>X-$K+`jDzBe$IjWE#s)IhRqX3yaVloRl0`>Mu=?R%MeJt;LYb-%l_pp~w5&#+J@?6gcf8?ZH3#)D%)%L9!XZ9;J~O1{Gf6}pNsnw&'
'dWrFRTYs7(38p8O>Ky8msu;+s7@_6tCb?Zd$2rTEU&yBh>h-AW6re-y*z}(Qs_%v$<(Efi{2<2<&N%P!$#4AG`7Szp5<t8-5VCg{E-?=ip3EVF'
'>bEE=sCNfWHuNNvd>al61V}F^FqmmXcO2M`?F1(!_^o2|(F&dlBs=pX^7%;3LZS7F&Bn}f|3rRd4@C88YOv7!S_Q@}!;!c~CR_vgp0UzIXT<c_'
'1$@R^fMkQkC$K{BA6dgZtXkA_L=SWZvqm;F%kYCw)@@_rlNZ*K7T3q?Wt8wbIo=S{n~Gb!Xs0Hr;RtOAYQy`qVS5o0ajL@`Qk$#3I?o4(Gea<r'
'#T03{4zGQlw@V$J6A^Ipm4yLu^crLD(3rxja8s4q8?yn>oeL~hU8sQu6#n4V!8BKOPW7V>GH$z#Chrlutm!#o^@N?M(8x_?qW|1Y!wot}NS=g`'
'!eNhAER0VhK)NR@Aoc4Ac&GRrd`$1VV}Z91B}?J)xXtHqgK2?2x|JRXW}8D#at^5WD6FV;5t4}ee9?;MDwO7qW~@GqaLDfeG07PBmnaRUPAr1I'
'92E%r$S0*jH0VzWgVj(!CJMS`e%$#NKu-(<#|U$psoaVhToo59hS@4&Jz<7kVg2$`@mb#PIL2sg8nszm1}~0Tf}=5E$MKFHr&%tqS)As&xLR?P'
'-~4{cac0L^Q6sZQK>NHF14r9whHK}QIG&?=J6cD4nm6M-mtJ3nn&M`*l#ut{z7^LvD#l6UeK4c~9Uiapq`6%N>*q=^+|?EY>_iA0e(CBuAbGCb'
'*{4*}r>`>znd)C!z(fIYTQTUVl)##SVg3LLekUPx_U&svmXJ2F>K<ZtgPxx9#7HuBFH3S<B<sy%#79iBoj&j8QJj%a4XE0dc)c&OR$RbevrLY0'
'x2tw3hGZ4S*qXlIp|__;p0qvlo(ODGzXm1=yHR_;-8d@h&1&)T>b9e=z8lo7p-x~#uy|o*C~iHub9L2)$}19`Xh_e(x32PL4jrlOt1XIAc}ZlZ'
'RaLcHokZ)sr*Ao})V+KuHF)C&?*qYy0VU;Hi7Vo~O4i|t?mWsV-UokdEvtQJE@sv6j73xniYsn`#fj$hD!uk&J#;Pr%DCZ$>{h{9Yx5LSg7XbX'
'9*|8&xAl?r-FCXUde;LmR<c{g>ddkopcm4~mF$7(OD_FVx{Onm1X-b~X*Waj#elKOU!nD8Kv}tbXuTOw%8Q5U!5>nNuzNJV8&#apc(!(G-<=5Y'
'>Vzn%xSvcO_C8t^GA~<%N?y!ClL&%ap{h9dzGyKBHSaqZfjrb{FkS~t+I=Z+0{(^qVs6d6KoM&79%$kTmVt%kyq)T1-b#y&&pn}WTljb@+w;6{'
'(VbFa-Sv)(3JgJ6<O$}vyelUEeHQoA>x1ryiGLR2Ty5yKKwgEsimZ1Un6f!L)*+XDpW&G|-4eH!`s85h4b^c1NO3y!P~r)LSB>9Nh+gPH*R?1W'
'3*Q5wWZ$RH$XC3p?Ur{EW47pdxdB3drYRr%mZ6r0{{m1;0|XQR000O8003fL6iyTy+zS8z%O3y$6aWAKVQF7uZ*6dIb7fy^VP|J-WiD`e%~@@4'
'<HiyGZovP6y&zZ;N744BX%M+A;Mg+a#&<SsC28s!1VyeaMij|%mpZ$y|9hX=7ilHYSNB_eNW@+4&dkp1Gs}1H-d$CE@OwS+s#@|%UC+I|Etb6X'
'I#E;)o}6)yKNdpLpscgB^v+WuPp&V#v}wvBOJz}4(f;24-c45tuc*A#o3>ocyr!))E=1_HJk14$#WI!L%j#-cJanzvh^6&{b`&KKy|l``iW74z'
'uB6!CoAW0Td1DA6cuOx@o^gl=MpA&(qA59O@@iSMb+zD?^d@{-x7;fhP2DP~)9T4v)OlBO0Xg5jd$+%*xF*T8lU>V`#4N^2C_%{u)@JT^QCF)$'
'EzI5~m9w&#*p1g1t`-TNwOp<#h&r2dxgK^CqnY*8)B0H!3%(k3Z3!+BZ`-=vo^1I)J1*9A|5Y@kMY6>HUe2eUS&+~P{7EXHU=lm@pj3K%D(f_l'
'Kh+fvhWmRS{_<3&UaXEB<=vtYzF8GSyvpjF`+-+X?N+Z4<TM^p@OUY>deW2(ZxGv*HoDQjN}l&x9O4?&|3YsDPvNbIpV{9AaIwKpqu<z&#AT;-'
'C1Ksc_^IJ+$Q<80AaU7Cpc%{d$sC6MJ{NMfEdhpg!z<s6LsEF%$@u%jz)J-r7Y*#s9V+@jaFt4{h7RxBb%rm0%y_eDu;|C}_<BT7L>CJjk~x=Y'
'QO1~zC_4Ng%DGao+9_hz@}^8P&Vsw)(YJ?(_v@`~T0qqZHWFNs_os|ldCx?SsJ!T~dM5q8483Q~?S%z;9?RBVlD63l0hP2JR8rS6^q`EUlfpU$'
'0%r86JZxH2X(?2cwtQLe&j)biC9l#d<DNDkNS9qsQRjx_GA*I%NQEAeM116k-}vs{dJBTOkumW<U=-E0D9E;nc@rrc>&3;?u8#_lkjwjlquHra'
'x8Rae(aK!Hw|hw4iR6#c4}$1{OP{IL$s(<aDg1$jclW`T@r}F@ypC77&;C)d==Zv)d?8z3ZC8}$fd}S{)QsOb1lS})f0kX5vyjO`mK?vSlx;ET'
'=)@Sou{j)T{gl9~FcNjwW_l0=#_fEqvXk4}(KtE18J%5>ZzK6ASrC103(56KA@E9Ma~3V?ry3TMyoFu0(ge~z&m*ih42*jy(?m_$!kh<+X;h^P'
'{+<oAw?xQ3c8dx+lZ~Rk7tJ}D$j}MdXBMvBU0f$;qw~v?@#sv&o~E^BOyWbf^j~+3kii{uq$qC3+CP&S3Rm4e+1Xt+L90mkomlVFNx}|wl<DEN'
'S5S0mmn9K5->EDqXT2V3|BU98oqUxFiB7P2dc0fE=}l1fi9p52r*(Hs73hR!*?sU|BvNs#rnFfR8KN$icoS=Duf}hZT~TrlU=pUW`#Rq&QgT9t'
'0hFMYx}J)ih*jruS8l|v@wK|0$Iik#Ml(rXv{1mYdL1q182b8<A|AIL4;7+G>bV*jQaZdgiJ6cy3XS+<1EnECT0dB+2RWd~q)W+3eQ_-!Gel-?'
'p8WL^Wd@g>&PMhP(-C@5p{A0wAZ7y$rnIa2Qy?Yz9B{5$T7aTQiB81pm5!uQ*C2&+|9L)K>iiWh$?-OJ`j&4Hk;4g%2bL}pc-d9LiZrFmc`&4('
'W#BT|Qh?pa!snM2CrcP`Aa;qXO&<}`C&}q0G74}*LbxGBB96#7jjvQ`U|A&VIj<;-ige2TZ$7*(?=I1pu!x?%ygEI(Om44!zBwHwXBRgyol5(Q'
'38)}kD_ZY4xxBpkHTn6|<;BMf<h<l&r1#&(Ou~Iwcyf`PUVXg2x*a9IUfuk(qDy}4l5^H+InF`)hm+HvdI)KzXQT1x=Htbui`((VDHxw#{W7}w'
'EoM_xbZ#B%EANK_NujRFCr{RZ`*kx(v-vTM>owg3<;as=Rq-+;aG!UXg76vC7LtauvhH#Z3AE#pQwlWxjr5PN5<NXT4okym0@l9Ijy`-B9pc{+'
'3l9T?Fu8B67QoQ-OF705?v4z0ARrk}*-Xl2cyO@VJ$$B(F9tJ|wN4>1De{2nUIJ<6Vf{X~Ya)dXS!$JHz<RvK?a{SbatGLsl)$r2rW3R;Y_NdS'
'b;}~FTPhxDk?4xW6P=d<ABOmkIfi6|!K^0G;hr0lBs>er^z;<a6p0ROOiS6lLn8Z}>nxI}$V`&J_+JQ@b|VGkass+IC_Mscx@Wo_pI!YtHfJ`~'
'1_15YLvNT+Jy4y(G1`}m)=#@KJ}2MlgWWmubXin9R>gBsBRAAlk@>;<?++btK7+X4_L4u_&BNXLWkhZ2vZTzjPC7jmss|>sOVz(p?>Yq@<~cbz'
'>VGSbk%UCUGwHGSq;%^G!B!F7Jm$&*X_HQh62OFuc(?0f2Ma*1^dKz;#vcam4i+M*1iMrxsCM(sY13r*2x~e}=$@{xN-N!plJmwtayU@0w%y(+'
'ZH;v+w4*Jhqpnf(a&nHh=nDp83!Vye*!iGJTix&vkcL%*&6M#d_dB}K$$ZLF>dQqR6z?<Zt-cx{)By1yY)Q8aIUjl}ww`4RK*GJQN|$f|idyDK'
'Y+K?S^?U^Z%p^k20Rv9555lf0iv@~Qo)9FUIfF8G4Wf@=Jfz5D2H(wiSq`Aajq)_(kN(jISDgsMH~E%wr_lE88~cePe(solX7|Ud*j9N_Pp>8o'
'aO%TK^!F7aPk9G*6i6?H>J6b?WFP6(Q+0J@uL`IsYh^J_;W7piMr%^OWqGg(z&Dzh-CKze6@xI=hMpAo(Wr+y@da96b**855FAAmRIjd4*wP37'
'sg~z;SLGvp#bT+LWg>old952kW0B3MiEGnJ;MGDW=LwoCfW*m@BnL))1$1i%$aRR978A`=o<dofd6zcDPGdn?W+lNLalDVs(BEniR5<Ib(08C6'
'GHg)@y4Pd<%HL>rUQzxtjv!}PFr@lt^@%u9>;)TImRtVM76{1_?}0G4@!NdeGDEWtc=_=>`FM3Ux}@f9J^RbW?ZwrnS6+_Cw`MwcdFgc#b-C!X'
'zt)o>3$@dS?BC{g6CRCO6<pJ@6SHl%f4#O5Q5VRXi=_mST~|4r;kyrqhnsSsORYO$Dr*UMKnhWS=l`TkqI+%{p1-v`>R$S7B@lWS2kgC3|85Bc'
'=!YiK_9<Tbr|H$@qG89Y_pi9AUhCN3*<mi%{2j{L@izrQnc!t?=xNL1`2m$9bmv+=g*C{dK@#z|^%_QbKzR^sTW2ts;_!#qx&tN4ZF3JC=lDzf'
'$8UQqX}ns=7Jn@~B?UcL)_7)LD_;Gcz2lZ@?Xl~Y+hXk;WlKwVW<g9{2Ei@%Hve;JsOwPE3Xk#vIoCk9Z7aRb+kPbI<8E=TyVcAo@UgoCd?=dw'
'uVcP&D0XVURnWH<eK`WKg2swKyVyPs{W(aREOp!i2pxd;&HoPYj6$h?a6(S@%M1Zyz3|w}+jJ^Dd%Z@2rv(X)M6GjyjUm^s?WLi6w`|vi_IlZA'
'|GLG30{z(nlc7z@))qtafTl{5Wa<*Z3<A2hQ`doddL7oU-WUc>S%*p8v^|G5lfJTyMU_U8rj1kuV}PXt^KHw$rrzE4YCWYCsjhX0I$4^pAgGjE'
'pFv(wW=TD(3}56g+xf)dm(5Ue{lofCC_7t;T5J4{LHK07Lti}ID|EY_b@uw^;5e(0_UiYGXu)v$QXRID5qWe|EKq$}=$YH<-D}OhThX9Dp&9z;'
'x}AV#5RgRP5zi2~JG{499CEhFx&wKJg1<=pMEfAf+AHpz)#|%R>2{90eI%UFy6whyHT6lfmQFJ0$vJO~vt2n1XJX*5IVh(6A<@@mT5kl-U4Vn+'
'+jYU7c(w9#T1el1cRTvFXSu<&&i#Z<U5`zH1ocYdwTtHkL7(G(e%rwSG!tbdFbT;`|Dcg1A-tR<s2@o}SJp{l>&K?;)O9ui;k12N-W}PJVdNjF'
'AL>7Mz5r_Xnc5_5kk-(9mQM{w)0G906@otDVd91LJEBVM`+NTZP)h>@6aWAK2mk;8VqLy@;oh?X002J)000*N003cWUt@1=aBp*EUu<t<VQgPv'
'aBwbgd2LkNZrVT)eMjOSM)bj^B5u+?RH_uIG#8aZ8kDrEs;aUUdkj;SU1xR;;qN<Z<6uRY2RxTEbLQs3U@%q7rN%RB^d3}MYRZ%+YhY!aC$j{<'
'LcnKggz*G2yD6<hU|V_Qq~GiJ9x4Mn@++_gVP|=bJc5O-Ob*_{#ru!Rx%@5?n6u5RtPU@!l2S5?4|xYq6;?Rv_cB%1<~q6?f%BHt5+T!sM*M=m'
'%T}0I8rcD(S7ZW5;v>r)vHS_O!C=tu6>K-c+IW2SnifF9pFJqIH`zhGQAUz*zjv@%cYu}3uR55yh@s%xRw(CHNv-Wx#PNLc5<^jXXw||0s7STV'
'luko!*uuYCj|*$p-|XTs=tG)*@AFBPwGwhHKvRLPR3?^o4nOsKAP*H_Y^WhwqDQlhqr02y$vC~5P3EaEOs6-$qY*^WFmax#j9XxMSdq+D2I^^p'
'H_Vn#!C_N54+{)!EofOW<{}lL9O()Bs*HwBOkp7uh^_e(jVI~V{oVZj@g|+!UyWz!?RYl(GQOHdf8i1$&lN@us_u{IUN<EU1Bg47bwd8Sf0(A%'
'lZQ~+26IyBKf7?za0u@pGKWBCL6^8EO&&kA9P4b!M)5i&X*6?Hdd%a4wPf6qt(?#~PB3xJjDo{!tvJ>Z8>{Dir@ebZoi}1ZMrqwOzJZ@x$}Agj'
'tIFqgXPQ2TX=+qk1h3rM!H%h3b5#ggXmT>vmlAF#IY#WRI@zPXi6cDaVB!w{bfL2KlEdYuwa@T~k?Gbr2=^9j9p-J!BvU3s9d}@|ji<cTPSFdo'
'WGQ1sj^Eavqj1#~n-rfjtiJ~C@`vn3C+LZmw_|GVM$SQWhm(vwd5xFR<Ja>GXg2|5MFc9_(%#Tz`!cNY37Je+Ii}5ruG^+YBX0izP)h>@6aWAK'
'2mk;8VqNT>aUL!N007|$000>P003cWUt@1=aBp*EUu<t<VQgP>bZ>HHE^v9JR$*`3HW2+DAphau!H_gK!iHgA5*S&Qs;gVa&TM6CRulpyoov<;'
'sgl%^py+?!k&<1-c8d+;2U(<!$9wOdo;(-~mRzh#JbRF8g9?N+m`b^U8kOeSpvdsTt^xyvb;&ZUP#7qrf?CPE$qbhQJ~r5(4hMt5=}Dnv1xZpg'
'rcszAz^hs++Ak;zv(ftW<n*Ll{h+1jwq#Nmd^9Cbw_CcEiV3ztwcDdVmfYY^ySd>Nb~$Qh*1ZI`G_{AFo@6D{8fG23*D_risqp;t1ZYs%ZxCw&'
'm?5u?29_yF;TqN7#EC*}n4;xI6IUPw!t(bil>j+2EM*!kTRZAn7TU}?7DNRX+$4#wu`EWe#GpIa4!auKMHst!eHU=#Ib9S{A6A>3E5DuSajY5~'
'0X|Z%61j2H-sT5Rd0mEt$P}0vKLDm7PkR-(Lemf*F>4I^-t{87iX(`xew{`zc?q+54AI-kGG4+5nQHKP4srB0hTFyD=4x>dZ=(AVq%wbMmu>v)'
'ZaNLTV9UcV`>ouOvn|UO*{3B>iI1++VL!-mF>+0pjT!&=a}TajPkXo=0S9R9c;Z;*^ytfzf(urb`x%Ittdtu4px@NGF$s~;^_1=so{wO|1zkxk'
'rqQ<3Po-q}ct(mDSrs*<=_aU@RPMfyS;+?iopVBJGmZP{#`{^j-(BMNXPeqxM~A3sG&RW<^MskPB{9@S752vWhWBT~>TH<D!&m3SoAcrFFRvv!'
'TvaY22>!3OCbMO<h+#5|=dGCjxSHNYOYkp8a2ddS2G{f1%jx7gCY1&7Vh(q=FRU7uQS2l(#>cE|a?C?J^NvHb!{0W=T1=QXRjqv@_!$I+<Wa)%'
'-tX-S4}05h)4vAZ^|J?m8BL?>76`mt%x_u_{(KcJBH;P>lK9@%9Ry*4CR>w6{KJwRDBFJr>fI0T1K@=_5KHnStE@gG=g$cW_uB#co~f1Y<N6Nb'
'HMUk_eNNHqMYh&?AG3C!q1I=m%vcGA=?$1Q1+CV=DWITtCuP|t?A6LR)v{=W6;EUp>3drXZjq}-02h>pliRuv(e@ZGIIN!<^VOzwyzB$7nA(v4'
'5)jqYr1}G&A+x?BjIypVhu?1Rw&z3UK3MMUs*s$83Eg^b`E+4Fh3?C<W(I7BG_AJ2|DQ!-zoc8KI$aPSM<la*ypxUB350m}gChpls;?&$yFECJ'
'YjyL^YYoPGZwvV3QD+1-gaqk1@^kO=ku$n;dYi}K6^xe@|CMNGQxrzqLfcTUUGV}SB0hO772bKr$>(on30o`kJBQpJ>@0JIW3SiuZx}^ywBGqe'
'qtE{UV15C*Nh@yu3<-R&C4!OkSmYGNR*Vbp2*Ep&I;57Q_V##ezx$p0xtC>UftUM^@!a8duLsrg4^T@31QY-O00;m80AgJ+iGDG|3;+NXA^-ps'
'0001CX<uV+ZE$aMWnXP?bZKvHE^v9RS^sk5wh{i{Gkyn@nTFB{)={z~`_w2i&iOpaG`(c*-252Zqk$+$;+Y~@5VEbPdWODG->I+AZvm2`DB8Vr'
'LjMp60*l?n?zbNx$8oNdSjrO}Eaj9vNMFZ_&6P+NfuAxFP1$`ANgepi7l{tyDCJDaXet%Dhp{ihaLp!b76ucMrtRaS<D>U6o2d9Pm5Q^ujHD78'
'edUw%S6WO$$^3XKIm0|NGj$sLhWEiz%q2_ZKUOmGW!h#JVR(Gx$FZ6Q5hOqvC0P`!r3lRc9W%X<kQ<~rh~}*0oqJE6H(ifieEi6!G7aVtl#U(*'
'spUA%@zG4hOEw<QR(hr6c#JudSZP=p#abBSNdK%=$>LVaC;bSYWD*jg7qB-U{v#e~-0{&=&X`Q$DA}MP&EfG8!}SIGd!*GWB{yg#qSRMGqF*oJ'
'h7it3)@+HFERqjWu^{?{MKCtRcviub9JX9yDaa0zrzG@`SgMtsrmbWW3STyy+gpb-=Rg1cms4uoP+OzivZa=0x#c#ZCN%C0@v*x1jRz5~WNt{r'
'Mr<jb8XjUGUPV*;mc4@Sy=HSyxtX(x*z4Eqbl~CI4($^?us`Et=ymh9PN}Jlraw~)JvcjEmLeHHA`;Su`Ogt2^QDY*nro(Hx(YRC{z|0?tP#7p'
'<IHP&_U#N!*$DWz7@UozE0`fv5d&~(_Legjkw10unZ_&SOSj2JBj!dzDHAJhX*gZk)U8o2zir2jku4{crc?{1E5qXmVZ4&NR??cxC6$fYl?erB'
'6IeP6W1)>-ZX6@mxoh&hRw#hB{lPF;u&)nvH3Cf93-(owAr%$qLMw$u5H%*eW<>LF1m>neUo%l7Znv6kktFEXm<GPJ1ma1(k@a*-)OwZIi>$dI'
'I~Vob5;aTO|DWrdCo7ObP8)fg-plnU6wAp}u*V^L+^yU$<{wp{<#-uWtc`)K4T7DaK^uZ#;A9Fx$lr)FISusK-kM0s(*mgX!GJY@(BzHqeM$l)'
'f{>mfpe~_OqT)GJvw)vSbz<U9E|Y~w1B>b;?t`%<MXaunKFok2K#T5$uUDoP$Yr1*5-LJbOsp3oGFq}wMsvN$LF{c-PRx`=!C8n0Dgue1qCsHk'
'-`gJYDG38V&@7(KSo#=fe<5>G2HWWd6D}o!apRCfVikzc88YX~Z$0nw?8+Mq2;?SlIL%v6zq!1+x}sLZFMLsSzcQDzy~!NiwgTH`hC4f3h{Bi>'
')HpXE4WhH}qFH=L((Lo~I*~*6&}(~Mi&}pS#rlIxe;JxuRciSaEht>n@-sZ~QLjS`>^jw0c5Z&^wU~LGnt86M9O@x^k2=cP6(A%k?x?JC_M?JX'
'YW+425t*EQl)#y3so{r}COh)SZ{YCa9**MdJB@nlbCfWI5p{(St%7@rk+=l`kiTS~zv43yK8i7C7tuVFBtv1C(YTlky%u5`KcZ8J8g0S$Z$iz3'
'===QULw@rUXYV2q1_(WLN9L$YITMssm-3VNX~jOG!tz0v@7A$;+c7`5edF1ScerPMI-dPA_xu$9`~2La-%b~QH7oMY#Q)xEd@t6-eLfmEg8Ea='
'G7s2wddb$=4Aq~rYt(foi?b}-M$7<Vo+{&<g^f^+Gj2Av$w)2f0xYEToM-rWB2;;7GtX_qU8s8_wCY<lC6we$i#gz;%xUP__1N(O3!B{3g94S>'
')0M56rgL+z?Al6ThE=M8$*;T!%$!SPb39*Ry*Ea~E#Fv*Vy7$~X@)J&V<KBuzS(TZi0yW}Av3nlp5Y2kZ<Gyw;%p5p<qTNsBOl-oj8bCcV4=<Q'
'qkh>H&w6eKP9~zW+uI~g8=EI+x!yK!ZzsqL@Aej}L?0c0V<y?6TgPdcnGWjM%vmxIKOx&ZZt+R~Hw$yOFy<76%ESvwN+|@KJ1ShA+~#1GSCDp2'
'>ruvtoKI~Dphc^FQ^CVL@KTCz_NNhZfX`0JF-uxM8y^)4XXe1B0o5`mbGu=4@XG|9%x6aDZtm*2k)2|Az;EnWQo4cC(=d-s;ygXtdr%VyZ~Ie>'
'NF^%$c;k{p_pWKBlW02zgIKrs60gX#Da!bX8F+Kjxy#rWB((PeX5Y{}%I$oADVe=L!nEg|X~z9rGOvd<Ln;&O`_WF8TU%wBU$Cp-2`iYbG@A$t'
'Yd@4ip%$AhSh`ARBSf39had$FFrC`9Y}Go(yDMCkt#Qd$`{Qh6jb04A%F;&wN>iD!*P6@H;y1}jyHHIQ3gPE48K#m!r+qQxnt<Gr$*WmB%a%vr'
'%7xrfRj#!z*ctpBgPzn2F<PKuXT?cpPjajClI$dq))&=K%1!*li$s#qYH4=s*+kE<b5OnDftKXZlP1|dk-7mXzO&{Ygf70%dVFGEp+n8aV5e{%'
'ufYAy#sH*YUVv}MY+bAh#tn{nzpL=TLe|Q+;U+8dylx}SgQU@{nc3oZ*V_kggWY|L-TfZwf4}1ayl3g4Yq!f5J*>)gVsj|oug?`Qrd9B<jiwbS'
'DrR2hOKU)xNYvVJc~gvsER2zKbj#>`!cygQp@Qfh;AsyO(g2~O!OXQ<MLwORlrqx{9p&DF7Hw1H_fUmel@w`AslYaF5NQ>Ucl%)OPE;_R%e|&}'
'wclxb04tX;);GHk9cXhZw1l4nOA|ow7pSvpj!nQiiH$ebSpmW-3Q&V-RbX~_wTvI6*^7a?ve@@mSe}#xU{LX6lzHQX>JBTkV`&)c3SKQKX<%x@'
'c!l<O^l7=qD^NtQvfL?VCsCss4-MV7KfJ$w_seyiuZHgRx9=D_ynpxh`nw<AyU*`n54>u}USIt6-Nnxrzq;F--C9-B-7uWO0-o8FabBXg*ne=B'
'vZ(vE^G(Ndxn((>=4Z({8+iPLC~^;f-432izSfyCbCIv&%WvOZB9qi_7<GHAK^-^Cbjxkipy~2%^BFj{pgyQWr)QKiq#io8f>);lehzE9&#PBf'
'@$#IK-GAzN`zoS|PP`74d_jT!naAH?Cy~vJ8D?VIFuRH_iRxEGSwtG}Luqb_QIh1|8w*OO9{o@H8M|~_#*7>}v-t-3d|A9!W+fB>nHuFMo^ASP'
'P8ks)e1g!r9~0(SDmS`TD_N`e*$aA8f*QpKZ|r7k(4#Os?OHR=&kK6Iud#A(-@6uZGc~J_o58Jnut#!p#NF>*J#h=g1SN%vWdg<Xa$!P`8Yue6'
'740+$1Kn`{=^8>&?t6EDoOFn2anoN&|2_;-O=pU9-cB1FcWY3Ms=J-NNyV{mkKn?vQM_n=HR_!jRm>xa%m!2vVGaGeT`!;C9Ps8RpBq=l2|x&h'
'O{#GyXBzXHDoTj9FJ}QcXC6vvNw-($-e(0oQf$o$Q5G4dc%6&Rz~xz9b$i*`QeyV7wugt4h3n|dwQ$z&$oWp6uUmcEzthh2oG(t0CY9vNr{{w@'
'utDfN@xs|Va#Zhjx&&NSr~=Q?CfJC1)$a1TI@0ddkOpIsD1pZ*Ea`PyRuZAtfPO#I6uR=9+zFkTWb9=i((hwi+Uoa<DYhmPd)@s^^zux!0r|(~'
'UUkhU0%g)IlEcsvN{ya3{M-=i0#_&qN%4Ab{XXm}F~M$zxxGQ*BFjPT(Nc9$Dx~U<@K*+bh`1W8X5BMvw`$Rvc=-)=+S^S>Iy=KZ8yu_zJfU5S'
'5`!)`2R)}C%EXNbXI`5Yyr5+&fVo-5=0zsFPXv3=&oCwL%U-dD(VztsDi2$mqg)`CGi05!J{b27x6stZ5;KR+v!%l08N6LRoeSf~UH=ZP2c7?F'
'9WTvu&)c@=(ne2#lbj?~v>@fzoOMpmy^}?g$R=p0RM>))x5XxY40Z^n0p5nF?*u)8)<PC(ji@Xgk)+uS#-NudCt##sb`YK4W300uZq7kc*>nCN'
'Bk)cd22(j+CA5dJ_ft?~r%KVMGStErfde_Trz^zR*f=?>)CS}VeLdx56`#@uD6@FoZ5hd1zH8ZXORmbglhEOF67xLFhQDaD*-cXrRSqP`c`YlB'
'7~Of_e6H#9knGua#QI88RiKCEXt8&E=<EuhW44ra6|j7M_dle*F;Xp_9V|W4t0XDMGgc?fo=o;Pa3zt$bVE6~&EU*wk$X5(&h2~WS<aPjZx1lc'
'DPvgaurWKlDIZ1;D~5ZeZ?j70#*`YJ!I{k+q-f^Xp1BUZ_t`>fw*h%X<gA}k@=%=J1a5h!v?HkaKle}l;X9mE_=vvSYuj)92%F6p^49I<6ujB8'
'-_XnKtB{ei(zBEE-0sRTLuHC*4<Wv0-rjeVw&U^9e*sWS0|XQR000O8003fL4{72B)(Zdty(9nt6aWAKVQF7uZ*6dIb7fy}Z)|B}c`k5yom$<G'
'+qe;b_g}&4<!n@I9B^-&bAU8W&^)A<CMjAV2^3nQY-MFpM@qZvuJ(WLZ%B%wKD-ICNOmo9INvjz899og_d?6^Rxr*Eq7sUiOk_n<R0meevPjP?'
'SMtP~qhNKr#{zrthP{-hT57?R__Gzd*{)U}v0us*r?Wzdw2_MGqioBJRkC464g!s2MU@N1#H~<gCTkFD3R&TdYD;11$V#kAt`77l#8ocUiPhMs'
'n=NA>kA-F@nGtl;s!AKS91}6qvQ?=7`XUomDq_Z~Y_;d<%^g?SRSG1ax5rso6I~5<z!PWs#LF^f;=YD>MbpbI_un80NbrWGykdK?YPHDN{><u<'
'R~1wggLst*Wj^vYE2JUH{`F$JilS&`L`ssp1-BweSOK-E1|U$1kzKD=ZcT~)lhpQHbFRq>ZuLcVUfK2fobjrGA>7X2xE60^CLqG=LX?>U<3*BM'
'GbK7biWB#k-Z=BUw~AY>R%yw#X3i6`SFpcS1ux$jMXWu6&Gm}m2a>(YbLggkRh_bW&5%K$R-wqOcT(NtrM!cG&^cp1{DUq%NGhvF@f6l0KjEwr'
'cfFrMv$X^u4FAe_Oit1PfkCDPaajD1lR-6Ht2YvZU+F?O;HRhzKi}jY>P<yaH^w?j^xSfTs^nk>t9en0lc-<|*;d)Mm(?;)jAAc=Rm3&Zjbb|%'
'xrm7jgbSO5WUF-1oIO}t3SL>Dv~3crGQTT&bz2DQ#fkAI4^t}vfp3Kcxs^pKBy{5UWuCj!ZC#f}3a-w{ZJ}FU>N(g+lJ_MC#p`2{4ggw|)~=3Q'
'eu8|7g&u&&cT<O`)kV6%Mt;Ar*qlp|Ae!@{6l3Ki6D24FvA9c0q>9o*T4@}vT8Q+R7`;@@HQP&B0_HWO6(ImzFhd-)1OvkW<kxUp3)nS-{BTol'
'cDKLw<_1-cBiJnAPM`4jg(#2{t}~JIwrrA|Q>r`f2x4P7cR5Y`__5Ur>nv%+eY1AS?0OcZF-23>Gm{mmK^;?G`0{1LuAY;YY<z<6N~en49a>}2'
'_8Y=zjqP>Oh|~5Unsr3Tq8RSI*)SwoMmrQQhKQ3yP>yAJ^w6jIy(&ch$Q}j)kI@$KE!5ge(JK}{qSxPJxoP37pU8GGStqt;_+rJlMR;eB;FEcx'
'AH(E~+hIFr7lqtnr>NH(4@`-`Pz&IZaR3e%*f0<=f1l5QAem|=U<T`#m1BZ3nPL9gQ**jB?unzZ_cMbHLY`ouAE7i7-w?6H!8&2#_cJUDLS4Y&'
'm@GxLwyW&<j{Oz~3$dX99I+vIW!d+(H`jX-X3D4hnf%?v_Sk#f2|@Doj_0$3%bnYhMz8ka;9ubOylrGOGk}M&q)<p`kXd^$C_P2pQ6-r;NvKUl'
'pZ9CK)M=SFs3b(k>$<xWd!%^LrC{f#OIxL->N2j6dx>JBOL1LxxK&_QUUqG<`Mi%daX<!HFl*DMC*dBueS7XSl1Eo7r1&luoGORTro*nS_VT_%'
'J{79lBL=e>+DxXcOa47-T)LENx?rp0Dm@~}cMzv5s#|EJIGDWE^~F=y6em&QLY$FkT9>GM{PGT^UDq#n&~9yh=?<A)RrswnFdtfA?x@H@R-Iur'
'_lQ@xtPY)7>B=({p}Rd40i}X5)pWSvsV?NiW=6l~+Lc+#^`Kq**onH(WEddA<*BX2M~R&SdBqHksZvPuJtDu*88@Qi$b`s$9nv1Z5o)e2Ufxhq'
'a1=c2vA8uxYyCkIinQb|G%TECB})m$0-QEZDSRv3%rQejOLY{*vGL{WFc&JZ)q*U&c3?K-D9Q4J-eLRAW;hT83E>VC!cgXn;@sCcPX+m`Z2E~{'
'pA}X=AMwMXt<f}6uPH1bU4niE>2`*|?nX}>4})%gJf*5~D43Q}xcH~y=(He2l$~Le07a-)vi?Kh@>8D+-1u>Uy`eUpELmeK+Y}^ItE@x&Em+=G'
'hd9Nr+4$;b9s4KRJXoCmSimO=g(<jkpbM>v?@!)aQRrMIY#s(`G!=bMix5K7=Su&+PoeAVAhc^&UiJN8DGnTGoUFckyo27U2)A9Wf)~2-PNQRa'
'QyCmJuE!#6(b2~2>(^W~D2>o+iAL{CE%5cLDMY@0MZvs9XDfEUAie)W%SxbaRvCDbwSHy`J2W%Cy-rORX*f{Gth9NW^ejfJP@MgXGoNgYLTLcY'
'qlz;QOTTe-ZMSR{9ZxD}LNhP<HnWzA;4YU3HZ$2%KFWt9vrIG2E1MrqB~Ml8PB#||f9%il&UR?$Oc+m6P`EqvaEZ7lUqpav^g+1362R%ml&>Sb'
'GXW2o^@p^0fy5GD3rvR{<`UcE{h;|=U=dD5<@<VVLY*ET9yhk|JaTPuP>*3nQ3AJrDwYJ95MAbWD$Le*!X|jkH9+m=fYt*9UNJs2WW?StwkYsW'
'-mW7v)Uv@p;IZM)?zr|7qw<V$m<AEqHge^7InwxWdf}p4U@<(hpsVo2&|W*$OgiJy1s^!K7xM4YG%Ut2ZUjg{6bMYeD2t}F?uKkIwkb|X;IQxM'
'bsVNT@``>P{exXiN`6Mk!+uB49RQHQmqr-IPsgUc05{jjCBD9|n->H}_aYi!bU-w`$_i~P-4DBZ#NO~Dr@p}AcZ408^P)B;ZuF!|;0;Km(X4bJ'
')jU}~Ji5p=#1?YNWRSGP*bIb4N1>!)W=Qsi3tMW%L%kD+^;6HwWJonu`JCECW|#7LIPh9b>0^L_)1Q@QJu+Zaw-x-esGhhF!topZEIHnk%?lFF'
'@@Zd&*gTc-+Ne2j==?rNq&@Uf$|J~yWH&-!!It92Kn)E1w5ym&RblG2L{{b57tn?RkvJTnTy2Qh+3>73cLPe>xFU#VY%UYIm{UAO_Tlw^%{`LI'
'y9!R3A*)bfIw2UQ*ua;mJzfkqyTPKe*ehyDLi?J{q*W!QT&d}z7S%17dXv2T<L$dYKD<ia{`<q5m&v;qAOB%TXZ_^NY1AjzN!U5669!lF!SC4o'
';f}Y>7<|ErCiosEG30XD^*EOWGv;`FV3hHG_|(u=T?Y?OQ0WIJ1{_*N!)FSP#5P<L@Nf`yJ$g)XG@N=0-%pgThi=k`#u#$cuZ|v*A%Z4n?0g2$'
'>UeIBK<Y9`0QYFhzVLd9=hOYcl`qvP^D^u|Xeq8}9+kO6HfUT$@@!g?&@J64VX7EzUlG=-I6*vCaYg8FMSopiykPC6Xm8D!3fDhdG+v_yda+z|'
'wRlY}XQP8r{1r!Hm((Q^L8;6e)}GqkQB;gmOP3WH7^hct3U!xe^a@XjBV|5W1ze94yyezk9YI!nlHr#I39igr``!zRb9F2!j&R=^Es9f(V!qJs'
'C7mW#DYJ39lgyjIp0gQ^JbGyzf|+{5O%n|MhtO3Xp9Kki1m%|>eIj&&ic9Y+!}RM3>_(31L1we1o&Pid4c*jk_&98b9m+0z7(jeAz4M=f^GhDM'
'TU;FuO=_9l<T+qI^pxG`DITZqo%tQ`mEZ61KVHteyV27C6u!FK4VgY%4ls_Az-}^kn1T&sh+Qy|m_oV9#Lm3MU5^)I_FFKX4X)>1@Hw7`%+7!6'
'2eEcKW?dn%-%zlljr(TZzu>*#pyLr}6h$;T<0)0*{#MXT^W1%BA}!m@O!?eb6|Sh}D*LZR_-T)eioyN!;5EI^z^rG^#l!Al;gt%aF$&NO!SQfu'
'B#RR?#Tc;pIGNzGQPP8j(E`9BOnLynGC?yWQ6DfK=ThKIj{C6b4^6^{#4(XF2owg+1e=W>kBXiELE+<h%I|4~B4$_%<L7bvxf)PL_6rY{tgeuI'
'krBCTMC}9;6dLC0Lq^xk9&LMG*l#!N1@l);fYE%@h2EgRO$0(k-r&kX5oq<picIh=do>W#W&&y7!BSCc@2C*raN9}czjVV+{6?5d_r%(&5-Cbu'
'm|-N{Us;4l`&QcA4=(Vok?b3BdW=RdC<piK<8$Le@c&Rt0|XQR000O8003fLV0@xi2MPcH@Ff5M6#xJLVQF7uZ*6dIb7fy~bzy92ba^gtdCgki'
'bKAHPe)nI2<I9OxNtG{?POFPDZPVOMX0Dwy&6~!<fk{xpm?CuqXj@hFzu)fS2MLn0lZ$dwKg1TW0CqooyVzYNilToAE$W>trP4BGzcjp%`oPky'
'NbeNOtD5PpV6`YSQHy$3ibmJGU?P*6?dxh&^W8j(qN}{Fb}UKqMmM!c5+-;1s@9B`Wu>{6RjID7?3mg%S{D9Oa|`>VLMgMyp6hKPZ`}s(;iH+*'
'hrKK}Zv1cMVand{qTsiMxUx$R886WaEAw}*#6PP{6jS!26h&smxJ=RtB$P;G?PTAALGKG*4n!0{vE%;OcW=!22k}cIl)k#UN(-(O`?mq^ol=dM'
'_=Dr+6@woo?nMo=OQv+qRxEnpbqU5qG@EiQHdSqA%?PY*KkWb;zUd5zdzpzc6-g~}wPXcc-5Ta@U}v&(l8Kx*g-&vwYS?&%60ukH1FRPh9|RgS'
'RF%w4Kwkxy<`q`k#q}S#oYl1Uuh&NN>TbzytE#Xos~VKdi;C;MCadfDV#+pRwZPwnz|S$Qk_ebHq6^}4)(}p_(k07eN(fAm7UvhDm|0j0xZWL-'
'I!j;@>G?T0!Z?$q+KW^t=n2Ulm@_eV<327urKa{sBFu6?Av67*lTuZ24_X!cAZj3CQ^FEiBBG1wRm^6ugZnd=1`hZJp<r6>L?KH7EADwM!5b92'
'72w4*Qr)2gaqrFKxX<6h`o@4=$i0G(-om-6ni^P06K0#7fx8<?qglK8obUH=?j&#&=NPL<)OA%yQ|}C`$k~E}K+siGR!R3n7`L9WiatV&G9d}{'
'0`XFFxF1l4nH5H*c@&%L7ms*#fU=EX2=07AQyB3i!1FkzHSjkRWwR4C=ry5T;*MoG<F|^2=XR?^X%2B>HaH_@uknySwW}|QjXQnA_Z?jF=s&Ib'
'>_~Q<SXK!|_heKBZ^aP5N7t?z<)S)En_5-Ck!cujR_i>n*Pz4(d`KsD<LJi1YoQ@ntSDH=Uaf*5I{oXQf$T42=IHVO`c)6l;<Dum-YaGoJLZ_h'
'o;qjf6How@ckY1n7>Lh0K9n^zJ&!|rTnY?#Lb8);$91~xIu6Dt<|JBXkQ~t>mnA&+9WU~{06@IQP_!an@rK<?1uC03rUI9tp)Od+jBxOB+%Rek'
'm|?=}2r+&pOO8ezcrAraWH#W2yU7u9cmp5VAM9TbGSgeQUwHHu*UZ5+KY+l*{J9l!v(-3-52F+P4FI^r;t2?h=~bf@cy*G48RAnA5RV$pRm?6V'
'a+FoQ12z6DGDiD^q_ANW@<n)=f;if;B;`1ikW4SK(%9U@s8?(T&Oz9jVPwtg5jO%zQSc(vF}iu&x~h8kED-_#-;kiS+Xa_GiV!<TA*XpjB82Zw'
'90*i|&KT`RRL{KJ%g)jJ0j(}}Gv=TRrCxWS97FBm$hqS^cy#_4bl&u1pz?D5B0s4flwFiZ`3ETPw^$U6Gtf@=$Y)1~^m7xx2LYXG9!NXzB>R4{'
'=BADKjpVagoU>Ge=K~J4+)v`KW0u(M^&zpt&l-cF7!o^(Y7$C6M(X){_irumqEwPHdw8CA{Bc5sM`?37@QUl$J<{zhc2{w@VO%ge8?$d#qlbOw'
'`#lz~JG*&^+A&iTj6LSLdCe&WMv#u@wLh`0rRXfj7MqjFmVfZ}vE!JfCr_F%*xUJY{?=8aAMfqNocAMsXX0x+-81Mv<$51m<3G>yhD6*~g-j3g'
'jnI=wHT!)bFw^GsAu+GHiek2cx*V%YS&N9+I#xWQ)Zzl9q(q=a#GVV)GbISXMrBq6C7H2;IbnvDK+%d-s5OJ#HX6N|wy|mC18~=w;yHF69h2kp'
'CX<x~1xgtu>U)a{#<F%P$t7tcF5#)>P=|@x+US-vWi3dL65QqtE>2J7@@v|GAkjfVYeSxy2HiXH4*BbD8t~ms$e6j)oA59sH*Siv=$gjnxb57y'
'D2WXk$UY%JPP&_a2OjqQON^V7Gp9I!M$e)3i<hx$zv!ke!$y?9jp}X{SVfHwH%Y3lYM99;^!Jp#h^M`qT$x|(u;~-6jzN5gI?V)gR<!KA5Yte&'
'Th0X1XDXil`XGB8Umxbez4h+tA7AQ|nBJ=XvoToMjmEvT-c!yjZ9S>K(#1AdGK|$dS=$zaNRmN@S1O9TOSZP-bBLY$MkiE4O<8Z$lubA7mZ$d5'
'krv--ZD?zKag3-KXL{WJ#(R)GsNVrJ?KDGR>|QUaX*~_|5of+_kbY865Z;PgvGiT;^&0oRnfBV9H(>*@DU(c?1~Rs0!L2{mjhLQmPO|^8e^n)V'
'KmKxRp+QM5I5zW#{ng%*=bDnf;RsctEU=@Zw{U^BrH+pQMZ0o?jmrtYuOyV2oTZ1f5b@{rD0O{k!}<=)<uFGuPVi=2_b(ND&EC>4e5@h5!q~Dy'
'ohQN*{y>QO5gsoQa@oWt44@-<$ZGxR8Hh-?rA#{|J6VwA9MJ2&C2%7XP&{5QUoI9mamNpgp*}rY8PZJ%5%X+%AB{1^TKIN+-mZFOAB5}akj50x'
'|MGqO<?X70vA$oWwg7yPV0~~r@@gEtUzyra!dr^V1b%M?>>f#B!xh~GqXQ4cn~1Xfu}FlNb)-!k8WlHFUjrr|F2`$u7Ar5vI6U*mR_|}gmT%+J'
'V|UPK45QrwU<?FUHfrptiz#&zJ4kWmQniccUp5I%+^A2vJQPOFFr5M?68Iga?;Mz@mz@1>>3A9>q)0n76ysq~)AS-K%7rptKnc4ye%S#TgYKl}'
'!I|fBbN3|>7-(|jngShrfIMPLX=jN6+iR{F90Ix(2L_r#U6X}9aR`T@S-wQ@djNKW4wLT&B<_*;)Z~U%2RYOEEuKoPXrE3V)ZGPBT@yN`7Q244'
'-Xst(5q7@M=TxWQQ^+(#$Cp)b^Z>poq-1mhG#T7+Z(q<V$L}wvZRG_D4mO;l>Z;$=!5?!DXe3KL>77M>Y`0&S$0HtWfZu$WJzn{9hsTiQMaiE>'
'M^f#S0HfQB9Eppy<vRv9qw0y-wLkQ#FKd7@Q4mcz1yy^Sa#6i${(Hlm{Ad#8`ED;Z-Dqb-Gb-2Ie|?z49$|{H9^VG>b{~KcvwQs4!4u6S{EJxu'
'B|A8f<8av&1x-gcy^KX?5E54%J>I{2|8^h)`#&v%%~{;&#iEz{5ViZcQqXC%Xv|H{MR{wmLnbt&Jph&yH2TcV$4S)aeD-JjjuJ6YNDR2MR7CN5'
'cD-0EmNysa2@PLgIzBO{cKBt~{XMGlesTT+#=fL7|350sJN2X4(ASQ^o+N7$wdF|e))TZj+Hx#!>x=p%#ne!f@dRd2;sYbn@BsYc9BO2m=!^H_'
'Dl7~igbP-+zHt2&P)h>@6aWAK2mk;8VqJ;M;G<9u006Z$000;O003cWUt@1=aBp*EUvgz;X>Mh0Wo~pXaCxm8+m72d_T68>)jmY(YE-A$U|Y8i'
'kWHs(fC(B*y9+F)hCoa7SRGk%MQJV=%)jqBJfuWQv^`0qhm1^8hv$AH69mD(>#oX0&hmzD+OtO77UE9*kwsmxy2-XewheFb*QRdR+xIN1cl%lv'
'IF!t0zikB*_l0bW>YD9DUhuCozE|s;x@(1Gjo9&`VqMkN9e~Rd_S?3QY**)9DHyyJ0Iz99KC7$pfvq1{3ot8j2b-PIW`f<-&CRB)?--yd8dkKD'
'ZHh*=EEn*5S5$~W273WDO)?9DU^d${^^T?Krfa)Kq^aJA@v5p@MUkA%%$R(rvZ6LW|B!WM-pl&>8fY^=YiZu(wrh*B|JAMcO`QoTY2Q6>w`H+5'
'n|_3!G|@inkyU#9ZS@edcf2h5x)d5^UZk4rR06|x+$;`f44${|-_h))__G7)#55|UXq}ONb18lm*E><^fp>%8K9;<q!Al|Q@>b**{6RGFESjJo'
'Dj;9DC};%Eo2G6E%tf8?^2eePS!;&UZeP|sPeBXzokn2;l&*PplkR&2fNN`!D$St*l3{A8nHg7@rW`4iK2(dwpot?z3CQO~tM;0#-PdK2J(%?d'
'Hm<Lmf|q)J^4XsqFN?PSuBhZ5NFMfy`ynD@HbeaZja?c=O0GqFp~k|%0^EoVuo;CFM6=l}7aNx5nxC+!+7#EI#d+9}h!+zzh}hX1wyx`Pp}7TG'
'U#S6-89ued-9>d925;Y|?|%94@t0rE)1QAzKm79J`9%=3AczvM_hKJL!$%F%n<izsOS`7Dr~Du<E<U9HxjX{V`t0K3?T5GNujiNlJ-<BgQO^<0'
'{uIn2yCJ&J&=$(RNKM2&vR~wB-jUy0FmN2aHA-}VeV|Xsz`V%XE7&?l%UlsF!I@lk)yUJpr|2(H#!O)&*;;^y6lM!Fa?Xq^P0%ADutG|hU9b|s'
';6^LZjH}g5&9%+LqGyt_2$;0&BScywO=TwbEjw3#(Ajc{W5efK{*V2DqNKhF7VIZpN)fvggceL*Z35bH!@+>^1$zV}LjWCJEndy%tEXT!eELLq'
'8%_hZiq)aQ76lvzW-f?O?DJ$!_!pbus~^~o--q+$Wjqxg`wG7K?we>Jj4)03eh<tKH~j{w+=4|e9(%DTuQpF#KTg#F$I-?#8llT^!uqbr+by8{'
'w`H&U<i=7pV{5YY)m9YO+t&LTO_4GmqQ#L25LSRUK;4mx$6=Ef3V|2V)4e@N;9>OpU<Ad`zj*W!0#Lk&J`G?7PC?josO2){U0&3|L5hMM{02iX'
'>j>0lv)hYnP!@`y$=$ZdwqZcSL9ZY5djOq_tj1JjX~o*4=_;3RSM(Y9kh5F#Bm!d;zMK{CFY2-~A{^M@aMhKinzr>CAKB)>lBKoR$M+x4Cu;!y'
'@ii@o5dnemxD(22EJwyTI7ze+qBtR2#`lE-0Xz~IDKNw=BzzM383tXuIr|ofQ#_E%pb-$FG7&^qXD{dT`C{cjn;4dYlG4XuS8B7~awh(Td<*Ib'
'0UBr;oXcdU_YMop7GYa65VKd2wT|p@;W1Hbkv6}D&m`?-z1}V;Jo<Io{sbA3xqxqrT+}H|=%r64N>z2UgE)m*2BZYxslbENxNzam(A1`jQDmYi'
'$cl3lOWgUDj%Oo?eL#Cx^4;FVCon3=#zm>HU~R1uaGtdgY?-XPCKF$icVG~(MOp7fAI@aUAq`qGh>IpQNz_;>L4B%R#@e~hV<O~gkok(8VL*5t'
'>5x|C8qj26+`Cg)WFYs>*Iuc$Q&Qe*cw?o)I^N<UtY?Z=>>y=RlqYMBDYyC5wnSW(5Gzu|^M|FTT!S!kH27wT#JIK!C__rniCI6YH%faJ0-w}-'
'teAk?5P#z77Ev?-Gq4Zdi0keS|L_2VrBO7qDYe19adC7BGJNtzFMKSfPbU=wi=d^mQ+#YaY__zp^sj9MhZ=6FXr=c9Ff&NmXqkGQu8L;vFFP7e'
'bwydw(yeohVQ348$!sSQlvpExwuVz(G=Z~dF_ItCX)3lc_at4g@<B2nU@!Qu-Z}>)x?MQd@-vTa+eXu)7tbu-_*OW^G~ftkggWUe)*9q>Gc}Vz'
'zei|#%mZ}2-y}Mu-9GBxc<>>vA5?P%M7d9h2K$Gw+vmI$%g12A5F&@Q@zYqN>|X<K&X%XJo%NB!)fUu0g)@35GpN?;9M8(vQS38^l;iU6*B*DC'
'wm#fKD9m-$u65ULRvdH4>Z4SvuE!&5S3Dowu<^emN>BM>+Q|0A&IcI2oB<W3Ok5jtRY$ehLMifPc(m!D*CRM_PowLhhtJL&ynVWB^Zqk57~z8Q'
'TDxFItrWV)VxrE2R=VN?Ds6Nu-MDV}9%=x2L=()CFRue;K=0rv0i`XUD(h$h^I0;s2(!OC8)0@>%!Wrs&?n{U0PZDgZT_TbaYWF|(G1WM>ZdIa'
'w$hG8K_HnH0lKP|ANE>8VX@f+ZiG#y8Z&^9-F{!HP8x3>Qb=zhjXelxqM{RUw{?+;`W9Pweh~cvKQ^2DLm$FaDsHvM?&G&At5nNUR}-d%WU&0J'
'^@Q%B`6~)lg{4)uTZ;zjwaWCx#=8(HXhh;3t;MPX2YeBhsq8dNyQu1#2BRCI3ab=LNL%x6w-51k8DJ;gsovB<h3NyrXZo$L!0^wyDS&ZHtF^NU'
'`aXx6eEPlmy&8HP!T;)xan-v++_7_44W}Ba)MKd6`t*U*afvx?U>2Z{p#Gw{MM$N%<yA{O0zR^Jhd&x@>0(m=AjfQ5*Eg|=(eSr~N#zzSkV-|F'
's@zpCpS2CINT^U<>BAOy<KQe*?TYOHB>~CqwgRZ?q}X(&gXbmb8CRY1QnZM0zXhXCHfFe-?ZG#I6=R<@+d43Qtf}0uZWZz>OHl0&@F%;yL|-X^'
'b*h@A%Gv(<XT+?KYF4WPWzR^*PGnnN6&dVnSdAxB3tMO)Q`C+GwMB+n6=SHpBp&y0nH7cZt^i|!3f^K9BzzISh}nz9izw1fE*-J{&hRJ;vlwGr'
'Y|AAu;PreyKLd0csE9%M%J-4;FF}fJ!Uh-1@>)m@4Os(N(`v*4%Ic=8GWdaQPH+xIi^pB2Y1}EjLvEP1e`0$CqrN_Wu2dSk8Tj{Vfe5djBFBGx'
'nO4M6*YUZ13MI>i@5Ip+ha6Mef%;yJu^rm$D{~y;I#KWrQ^(pzDAIr9TAaNjTPV({S2%_jC5q>ExQjKVC>(X>$h3&mk=HGUcnRU)Sz`L5Ui+C%'
'hG-v&9FfcsVV8I!Xk_9X6Fyg}cxVezpl)qs?s$XsjeP{7Md<^yu<Dirdkw+!vvZI>=hPCwv=f}{wFqCv<4%4C%76vtORMYMAOFGLlcPsB;cjdh'
'nrAo-H4ic09U~2LKU8FmYLnq8)jXQ4rT*(HMi7ORyetqgIoAxW6V9HAn$3BtuqDncOHqY{edOe{=nt(;lKRHYVH*x!$~jq{Rw^5aThy;q(O7Yu'
'&Z%s-Yzy*HzdS8i-~sn&V>BcmDF&7>dU8J*$k9NFZq>I@FE+Ubfj8{sI9DHido{sJP1&B_+qyFD-#dv<01<R&+UtH8{n8E3#gswz12hp{Jya@p'
'<XzEj!$1Q2MGzeYAs`r!fgJq659;Yee-0m%zuE=OcqAEoRzo>Wj#nKj@jG7h7eSJ?PTOrGcs{mX|J?-hkncTdkI!@huBx1L2EDuWLK&^qUf3NS'
'jNIcPH0`~QcoxIl+Ys(<7!yJxfvcdzV4C9TT(F7`;Fa`+^E3E4zF0DWw)D_zQMwTvZw|?&!I4gWMzevpw=<Ast}GDWf|UwS7AFtYJQGj}8NAD1'
'_ma)j$e+(+eDHV_@`|fKGc+x7?>etaMIJWjz->Ab1`+oRQ7>{*BDxM^8yeB})yZodIDIfTA{H+C8&_9Jl6-dV^aZk7Z-(?&a<Uyb){WxK;Z2Wb'
'Q>T(N%D2@bKZb`eJG|M68QR?B@tBHBm7(<cdNSOdN((5_=x*O|Qlr1YCgJ3M3}rOh!S>uUeLfg6cXM<GbL>u5^cm0{3ku#7_NskgNMiD)0PD%2'
'Qp`A1@_+_YX580_3{RS{RS4DefK@Vt^OZ5To=qP44r#RAn86E3*o(5qeOZs0Gc>9OD!n@MnsT-rtX^3PUNW<)EH?os`MD`KMIWC_gskKmyy(d9'
'PGZ9~_-?txp6opN?q4xpGn&&G={M{vx8xf~hn|gU)zaVJ8ZB+>mY1|t+5uynB99n#GDi;Hsr#&YE8Wo1R>?3m13>DWUne@Mof264POn^zPK6Y5'
'pKvl{<Z1f%wmNxEiHHs6&D^hCju~+^Us3oS$#qg=m>z|L&j9vWSqnbyUBhV%RFQRx<jEL2;rDcjH=VIW=G+sc4^&m8>ofefg8af5`6p3rb&H;f'
'7(TX)r?XCv)_y$689tue79R?y9;^=9Uu0?diTjfHsI%!wNWqJ!+350v+V&X*t3umy>%Gg?RclcZ_UzFn!dt8oDC;)nYjqoQS`RUGXBpPaMmjR_'
'{s&J4KLh3Oz4i5t5PQ?sTw_=;?Pl$Woa?3E{BuH(TilIq2KXov_bpO1Hh}@oD7bm=Y7HBB)ZMYQQ-$70lPw!Q8NbZQ6lfoIkxvA$Z&d($MFH&P'
'7Yks71aJY6kgB!gcysskEb<cSW)1NqQ;78@pX#l~{~h3@;12sJPvIOrN6TEqKSj7!s~F?*3E_UHg!}qS3-?p9lWG1&H2PcDN96#;*=IFP4$E(('
'V#h2{5Bd5e)0T)^riK${xR@NmVxsVem>M6`x&DgP{{c`-0|XQR000O8003fLbyZ0jSq%UHWibE%6aWAKVQF7uZ*6dIb7fz1V`yb<VJ>iaty*nw'
'+c*;bUZDSh=l#+EZq`YYBHLaL+bh~dfxX>gTPzM(6bda-HZPK>l9Ur4?#JKEkd$b>*iPC7y0I;CG#t)6^URR5+3fSS78y%bWi?N&VY$-Gt_3Sp'
'$_w`91KY?<sOPCFYh|RBs$zFa-{yt7JAL}(>66bewp5uYm^6$R@H2?Biq%@d1c9>P|7!Uk0n+De*;o><MU@FHG*eZvWjswqZOzjs8(zqaTM!{D'
'V|kSdbIMrvj!JA`jwmH)$}9H%eOZeYXkV7R%2**Q0ZPqgv!_pTtxA?8d1D(bl7vanUt5?|Da&yg2DPVAc3aD86-K<NwsZEYv_kWu9a3*IURjxj'
'LI31N{0lXjv-eUI+1%-uc>R^enNOdj1viGhF-F)=s?n)<tEybCp0(3YUOj!n;6Ji<Qply|di#nkl`7aZ`>Y$`p4Z}D6lUy!yyr_kHl?Xd4X<Cf'
'kKXgb_-DBm!l<Uo#vb0uTUm-s^4*7w{lMyGS%48rVL7alvo{|<yk-?`HoA#sw_@vrl>9zPVNYl`O;QV;d~F1-O;#2~qSc;dBIiwEudn88C9cor'
'tPt1di<8)Re=3;@1-t%xO}jR<9a{cX*s;H6D$i$gHd`xoo1k;GUqG8gHC8M9C2xvi_5~E7@lKEx_x2StR<r-H-&7?AYT}P`T3R8hm0e$-Uo22t'
'oV-CjA2$0%G`VO>m<>_Iht@eeU66ei_!lj7x@)3O`TYZp1G@(kHUd9G+*dBJO=~Z{T4`YnjvyPg<Bw5yae8?<qVI^C?-&dJ#H)IjNr)rKi;oI?'
'ZaZ8Wg!wO`!gb020$_+R-hr7a40xt4{Zoo!&deI4w=v+VOD#oC5Qf4&6HK);xfTpwC6^%H3Ph+3#DfI@jo#f$V-YZ{s-G7CNCmsOsli^t-rUTc'
'xSN|z@Y|akhS+p-WAc0WoTIG_;9tpQ#Y|Hpo`~!<p<+!-g@;BJFxANdbXx+j8Hh%^Mtm;eqkzAlqSOGZh{*t!+M3hWGGSe~B2n1{FfQ@~_l$<}'
'#!^67UfjZ>lC|I&ey>%Qa%1r$M6A#oDeh=o<E%WJM?y>rfKnxyC`lH;LDr-+H&ZFNN(I`HV{Hnx!oMY`#aCiSNZA;flJ`}zB!~`=oe!d-?mKQZ'
'>WYe??`|Ux59=u<N-6MTS_oc|GTVxm?j0Mc{X;M<zAc^g-;y>qaT_usp7vXBL?XO5Bt}A5ry_xA21m61I2ushYYZiZgg2Q~dtQW}wE#p<>0{a0'
'p{Um9M&vvc7Gn*;Nekh3G1*AdaLh$huK<m6p`E^88}QJ9z*6DvCM(VBwM_SjGNM5LcB@|t&~dOyyo8yFlQ<&CSlU}utEtj8xQvNCVW1iM;Iuv-'
'x$xtPpa&v%Xs-NzTEZyWMrgdxQS7!53B*<op60-_553`S;QS%Uob9BXv-1GOT3|#cgkO=mrAXHhf2ng@x7W^&T0%sVPX5;+SWjjl0Himd8O0g%'
'=zypd@b(lywT4KOmVup^xsx^pKn^Ggs{-7XxRrpw5xR+rFAI?kiHfQ%5RKO?L?=rN3I2-`ElLEim5>6&ns1~EILk@>gzOxdq+V-S4|$+mtqb@^'
'$ac#$SGOYUEm@`;;2FhQl(DECOjxQSi;;>XbR#^!(~2A-S#mu_|FlX))`0MY+-@-7c!%yp4Wk0w$!&U4@U75hoW*&TLOK8J{G^-P1^U>uvuN7Y'
'N$_=ax5NVoLeD7qEr8A=_TI@VQ+J6G;85ArYtEg1FWj46w&V-LqbF%X#Q8<UQ(v~^%~{3=lvjC_SrdIWYlPdniB=R1skqj>hh>4Z`K^<c2W$G!'
'VYmAVv?f~#|7(xnw!)KT1A}a5unFRM?wRNLX)sz#<vJi6@h-@LgtF2Qks$!NleNoS9AbOmV!>=lE;v?fpc9zN{?&?Za~lcE?_Cn~jA17W^kA02'
'+B+c4-N|ND=r|D$yk=j%^zD%3P#G(|y+)08;Ph&Thrjvoma1g&2P{SyJBouJT?wV#5MTKML0xXy$KOAFW<QcfKXU0y?j5{qszO5T3uZ!!h`eFc'
'i9DsKxm}Nv>}X7*i^(BCd<9spOfcajGMgOe7c(4kpa8lglyEYLj_+o*jvKc5?X3p)5Lo)l3YjovggL`p4E_vuC4&DPAGyYbBmT&(L9(h=X!BAL'
'pq$3+Gs>7R;-4dB7mI~|auUE7Ie+B-L@E<fq2--G5CL3(Wa<$*>{B5o^-_i8+R5<S!1WQadU5Gv_0q@c+0~+pRw@q;iCA=X=1It}S%1>ry<OR8'
'GC=C80g``=PjuEl@9`Bc%t2<7s;090(ul+67k<_c?4Q`>fbhyR&+T)E(e%d<|JcGFLA95|29lHHQiFrI=G19$?i@=5ETU+i2_+`-LzKxbP6pFm'
'QX30czq&h_VZ@HEZgJ(I`0}M&;?>1s*ET?DfO$+z|LJ^kdR|8i98=sml<$}&UoOxiUo1R5()q-yz2&{&AeIW6WGOAO<n9G$=T~!f_7cQ{PWY~u'
'_!9WsR;(U~lRmlY@_sHLg(Mu?YGbs39waS34%%;%_&yl9)3$J8_cbrNv5=q+jzl;2GWU?Gt1(j5vc$GP?R85gm)IM(Bk$ENjXxm6k7@xtDn9!?'
'yGcp?fdQ?M%646?$?4sYK;@7S?9FLz#GwY(;%cXXb><sxw7*3;@`6K&t|gl|om$0@Z55(`eH_!X(MP|-N%^_g&VGj9p}a((oOw+^5DWeK#8ZH0'
'P_gGH?D;>~tj~sKue!7<p~h><XQ=%%5YdPeI(MLt5Cx3HoJKTe&f;Id{{DwPML9WPe*-j$UPTzpC6tAKqV#uK!)j(9UBepcLvW>H%g{h7XR-Xg'
'=s5!oBb9;~!g^4hC|}}nNTsZ^JvIOb1O9&P7qyd{4{07t0B5#P#IhB!BFtdC+D-LiN;=xYuCIp-rBf6_@D%r)hDA+@gZlME0-WQC3$|QYCx(4`'
'|L<5esLC5%(VJsJjr{qOSYh%Ua%sO6wA_9q?~oWjP~d4tZ&*xMz=j5knbE$AYeWI>UTzcDbXCabMnl}BN{azq>LBP=q9!eAbm}1+Y|Asp1pR|J'
'o36m74?QJ#TsQ-F6HzcnJJFBhC@#Uooma=(bI56;`Qttv2|L1XD34DTVg4K{-segF+2L+)*QmIls=LHKwx{m+TSU0`y5%8%q*gB)JbEj&XR*9h'
'wd){Rt4iwQh@46e9lBFBFikx8goXK!RQuM2^i8~x0q(u`4@`?VXS?St=wf1yC%aSgt}a43ao(F4c<#~lNMAsEz7Scqg7MS<=b5t<1Og<2y8E1|'
'yQ=#Z4DP<QW+$Co>w6<YC#{v$d;Q^h?7TJjrC-Bk;QTdD>EPoa7+m&Ne*ou7I_FOO!PFGSEurGh<pWPkyXh&i>kN-ej;KX@`iqU536vubT_Vup'
'0bQG}HwjpiKz4$%l6dQ5Wql`*jbUpHWq089C|srTf=$Efk$4I```u<V3HMO(ZmpDAyWWRPZ{0a!?O+ld)<cx>O}x+%?3#76FFr)}L+7K<pZEH{'
'XU+mwIvZuigF6Wy3tkP4(%mB%$uDP97YIf(#&m1u+pY}!7O2K2cYZwDs{ByFPGN7dB_^*K9uUyc4ba;XtO<F;AFi8ig?D}Ew4K<>+~3w&fj=9O'
'#eK|hl_+wZwyW(L*mOqWde{#(YS-GQ*aMhdxjc&X_5OSjD(%akzAxHzmjA`IqQ^Z4&&k|PqUVlf;$13O_>aDU&C;g5Iz=sG>{*4YSxUzy?nNdZ'
'^>#Kz#-Q%NG<GdD9YTF=6d+Q{-M17ka=9LUYF&GLEz(<DTR2puz?G%IJ6E)AJPSN1%>G_X=mA^brI<1L<Y^P6HduPd;Hkjt4V_8C@19(_6}d6r'
'gFbjXjQIggiq3c=x!$|z_$N&rFwO3`#<Qw++rl=HurT8Yzzc;;l8Kc1p<}4`U4!7eq{%3PzE746!S&5~@@;oXay-5bgi*;8Kt;3bKW4o+i_Uz0'
'KJ(0R@UxEyX(z7$y!qe}0T3b_MU3t$-Trvgowu&<tkpM3Sb~89JO{^$O3g3FV10mk>ZJNMRFD-js>cvSY~vCi2m%8WdVsL=i#B6>))0j4(1-^}'
'45u62ajtP$JP7V*_eaR?IN9Jn#4(4vBW;iF5&-EKMe$>ddBiC}3va;i#p7VBcNg^!cP}sd75nwW=yopX4?*M8fc79ZZb-$Wzz=|HO*XB$wOTG4'
'mvbh@bX|?3`W3_6+|R|k-GJRXe^|lqEzP5qbh*=km&=?73u7#WCkkJnx+8rzoU$0pfx{Aa^a%gk@cX$I1{`Yw#=EC2RRLMO<Pmgt9@icbkrVIu'
';eK}x%@#))j*fyIPz8I*91zq`&SXjw=6F^9z-@`)c~!W+wku^)2o8&cB+gb3pZpa{J8&C+OXo2XAW|g$Y#uuK|4>T<1QY-O00;m80AgJt;+d5m'
'2><}v8UO$m0001CX<uV+ZE$aMWnXh;V{~b6ZgVbhd6ijNZ`{Zceixws(9u4S6I1k9z$U@i46<>IV1cY{#KPu584x{Wai&)!o9QNHO%(t4RQ17Y'
'sKXQ>k$u#CeBHXZxcIr&pjuV|*PVi#)m$mw3RyE<%2u<lqCLnC7sxwN@0nCg-huk6L`$Dqb}9pw3tq~72l4sY`Pub>1{7qiRDou#s6e;8YBW<&'
'fMTs=EiTbe;u7Szlld`b*9V~)&w11061Z=8ooiZGouA1%gNTW$0_4*|d1N~%qynT@=b%uInEewn5KT)wS}tmoQ|v{}OD|z2rOI)j1&u;}l&DV*'
'<>ek~Ky^`yhPMaOE|XP-yctRZ_mFishZ`;~F3!&iB`cPuMc1OG(iA0XB$|%bwQP;aG^uY^<-Fxt$+ZUUm-R#EXKv)c^`R8I{<}gE+<uY<r~KZ}'
'X2yMd<#*6C;vz+f8>u1H8Cr|`Eq~=2zQDeMx{cW16>knA`&{xmV!uj}0XdOAiO~7kUq-^2KkT!DhGRT@R&uX+wK+dy_%yg<!{qKa^qd_j`4N=f'
'uu^C@!n<5ZJHY<OKFS*Gc*UCrUFDb3PYwlZETdE-Nit!m;6EJ(Ok|ivjzPqfl<meCB0>wO8cFGu6AQib<yd2|Xz^cQg;2EJMKsGh8_!4sSJW6F'
'D2WL=mLVg?mF&phCJ%IMJWWJJ!JH^D#vR6TJfudFus{|?Kmmi!H=+*Hy)I5XdN3?IRnN^DPIV+>MeJs~vis%^h8jZ}yn|EStDN3>mm-n(cJ&9P'
'q#`2|C7a3#d-&!kquio}3blTL>^m|LS4SieuE{R6HAW_)H-@NNB?AVf23A6C^on<!qSD5oUJdld&muLfH$An`O!ETEf~Un^u{Tdsn=Z#xCMS_~'
'EzOGGhi{_C1bzMbYPA}PyC~vLiF^+Mqf$Y{*X-S3r+1H$-b}?-T%a^#hlRmU%#vm`09HUgP~AcM6)=F?uhLgFPdJ#0+iwxBiobyGHBuebeisES'
'Gu4gBEbE%1t=qhUp2L{2;dJ9?wiqpBHXnj6&+}0f?mE=7#esFa9^%pyi24ZC819gVhJH+gI$VuS=@LTf(NK<N5$(|l5w~zpVH5m&9pE2dW8yw<'
'?3?LsI!2+OzC&01oWB3`%jZvDKBU*zU(#Pb{q4g)f`|pdI65+pWf$2{ePXX!(AzM;FIVAXec)F={vq^I#ZYJ1eo%|}0Qb4rqb%X-W^=XP&IrfV'
'Ew<8-EQfARu2qKx0`9Sbrt;Q&&P5>uTbWRpUcYp`j<Ih9lQ1F2+C8z*X(39KXcj5D0mu$W(gd^VnOKgRW7sv=G9Y;vSm<EGZYI4YB&#RA{UdfK'
'%dUcC$@(<0+J2i=38?w5G%EQ$BN1|dW^RJfrV)ACgfxPpE6ZTIcmM#&BNa7t^de`B$l7ewg{BmpVSB=x(^9pRe;<e3b5dwlaP&$3)+jG1cjjri'
'!*skAO#}Hj5ec#xln~1X>yU0$Xx+TR;8nX|#ki*VxvwFU1iBKMqa>&cY*B#8hN0w!%m}F!PAiVz`;NtLUQle7l(iiam4vFn#5FKttkn<fUk6!Y'
'dpAO1Kz39e^?_1<rvQP00i+Gj;4ON9Asj{cmHQC(Fd!Z%`XSVqpjV6tvJ6A7$}1LM{kUSk??bO>G+AQwIckx%l)XxNqQji8e3Q~>+L{X`3@yKF'
')ngAw@z0Nfkkqb7NAj-MW8*Ivz8G1)<+o_Y2TNrWUlrf<TWe12F*;KQO(Rw>L~H7Pu-tOpQv?q$6x6aB^%^@(uC^R3_%|6juR8OT_e-a3HX6o<'
'G)8JSTLM=TG#c4BBS)xwSTn54I)&{ACk$YmwJL!;?x{m5Uhg4%r0Ek}nY627*yv+pmNPau8UuDmQQAk6*}t8Yy+vEZs3)G1!qIUcwBRAdLz!xm'
'b>zSV!pl7I%%W3g-y+jw)ZU!w(mfuxhzh2lmUV)GFhNV-(qYSLUa3#r<mG+3kyz2KZ=P=}Gp<H#&nL7-_M{%85siB3i5fq!_jqh|by+D3lL)kz'
'2qau8(E%4slo1^Dbt}ipx33XIX80LTsUlD)P1j;U{F@YY*y|tbrW6J+{Ar87{&+Qu1QW|%!O_t#v@SeIc^f<r;~fe=4TNVQdyw(6#DQS$D2g!q'
'{_2T9GtM^Jn=@br6tqc@;cqlPhFvWxvo{HJiI6Jo8pG*WLK5(d)&&b>Ul5-t;T>T#Oa=vowh=cOn!DbW5pV2=fJ_{0#%U6GPQhlv7$@u@&?aKA'
'aaMY=%(8wH5TCg*1G1gtV=^VMIZ8nuYzPR@+iDK8T#;MsUVXKC>7mA-x?J_1tX(WzqXOsF#Y076m-9hmt?wy)e*a=0d7(coZ*=BZ2(;*eX#E=F'
'p;L_(92`({@igf@vLl89y^TD)95hdc#rL?I9@FY3vv%HG*9gcE5?W0up~vp0dl)KBn4%aon5}obWW05ED<k|$J;&T5mi&lN0Pqc?i{H{N$$tGb'
'+``a9t{`lejBb<Yp%$E#qM=6WZ<kK6I4!ubZrrlpVWo1eT!7^WPg+rTFl!eP+Zlv-@3pohM?Tvku&%H|lXu6n<c#g<@nCn!5!c1*t0fsarEK<u'
'meADS+gJSZl)Yy<U%OjayKoPKB6$;!uS)1@gYR_cM7mjTFWs`6H`^6K{VCoYKZDCczh}LTJf$~FQc&x7gxa!zk1VF=D4@*!p64WVQcX(U^Xz6U'
'Mj3*=x(#_dPK@u5)MSWM<A_s7vUdsP*1MB{TRL#s+sDNyL3oN}E*CgFbCmdrS_|EwvIpKGqDNa{)7RK74FED#gf9O}QNt;^kM|as7|@MG7B3fE'
'k>gQdOX0VbH6`-&H<zBw6K#{W@MveY*z!gEY%<duRWUMy!A&zM2`r`N_9z(!4cPIcLrv=Eb#b3!O{?1{mE)+H4HjB~0kfdph!RDJS!Zq<^W%2L'
'l>XV!rVhZ`qYOb)biEQ_3UP@L2z_!e%?+zD9RlZ$8cs)$r}kc&NSU|qago_TYB(CFKCmRM7wo(7gB9SB>vFGbG?lPF#kt(VVEaE%O9KQH00008'
'0001DT@bsJ!yW?w0B{BX022TJ0AXogV{dJ6Z*yf|b8mHSWG--dg;h;&<2Dez2grXIs3<_HvMei3Hc0EkV$nl^-RuYURul#;jbg$QDUej0wCI2D'
'3?+Nhjf%>KWRk=8=DnfDNs_#`m1@C1o3KZ(|HK-CQ^xC{t$}Jl56aYFdmdAu>UJ7U0AE>$Kf}%rrlvh#-hRlFB)Pn3ob6z{ZH6#7+-^a4y>)>w'
'j17u0{N=?o*Qu~S4~Vn2-FfJP8>(;~9@QgG*gfrP_3i}xDT45)#^!|#!EtwSIfH{X^&MkoP!BGqaR*InRgia=7eE)~{?4Ha2-fKb%}PJL`v7XF'
'wFPa!8eDj+ELl<Q2XpW_+*_#a(GUt7;8oYR=tmP6k@@ppIWjBn4z+e-9Qe&wH(&<%q<z@hLmbL+52Z#c&OE>d%5_l`+2l+9B|gOJ%@ldmVIN(|'
'ZI4Efv?y3+kQ{f31m(dFp&!DekFRke_ZTP>)p;RS8I;+?N?ghP$NOIsdQz>RvR&`cd#n>FzkviQV@v626U50&NF&CJsKz?5O+!>{9%*<$K?HI0'
'NGM+<##eNeaCgrEm2>J@$bSbgr1^$6qG!-)BPQ61@TU>~dxDre5g1s;fY+;N9<e1mDTq<YoUqJ`Qj)xBd{(>&l9q1eYY^SU(*~OjB;DY3mCWS9'
'*AMV{(5%3>_Sim?0i;V)>3db@@O7^}&{sk4Y=z9*!Vc>UxHb=HkP~@Ch#c_tw|t((SsOeF6055sza<H)Rgu&E%{sr~(fTI8y_%76+Dls_#P3}1'
'uv|;{5n4KMaZ>&<zwvBjx`F(P?c;aBlZWv~uW3pILRw2gOw)26=25i+7sC&oI3twlgpqOzsY`>5N5k>a;2N#+@|y40QZBD%irk(mMDc$mRjaE5'
'ANk(){GhI)s*wEsA4NN*G?R|FDT*{KuK7+$$#DUnda(#;S<HBht;hf2p;nL&Mw7r!ELfbJaWf6e)460bDI#CLhK(;6<rC1<Vr;L}o~naLe%X_9'
'u5YKFm}NZdF@^M0O3B1!Ni9jr@1#s^<>++4hWZM65xGu14e6i2#CCXj@efc-0|XQR000O8003fLJreJsJ_G;&<p=-(6aWAKVQF7uZ*6dIb7fz4'
'X>)0BZZ2?njaA=I<2Vq0N8*35`r;IY9H3=y)l(jTg}q8FaJ#@uRatH_X)O}lXFCNJ@xR~LPTIiz*ozQNCgYiJzWHV{8jVg>D@!Peko(%G0<0x}'
'Jc2yRj`!~`(OQ84SwKo^#l4>e%?)=@Ka<L+hPV}wQWz+lGTCS}+T96e#;Vq}1{fov(aJc&rBsePp`_j2+1=@^%VKk@f_-<Vgo>143nLc5D#IJF'
'iICQrw(zG@5^dwj?he5Vb55nz2!OPjI7L7#SB@Cqb?5mTKGeW1kS{lP<T1RrWC4{jKpZ?dqH9Da<TCU5eb74#mfRXHEjl$|EG4>K)S?)W^UF!='
')J^0}8FF$nwHdimL~~IRkfl~ax-`Z!&?pC_P$dv}f})L;tVIb5i}iLxR#0m&DZyY>Ti5#rtg3@2qLe;|b-Z}UHg(6Fjy~kSSVc?dz|Hise>H9R'
'5(cl_ps?QOO))7s70#Z>^7Pq(JyWt0%QTkGj@kX?g)(R?g~!J0FCf8$&i3>*SWnH6h8(+^fm{ot<cyw7+1d5io9plA?0))vdVR%yzhGalKc8RH'
'd4>%ZIvIC0RRq-%Ys!kY<e9LHueqrCqK2gJeV~>L?CiHzI?=$nG0G(LQWe<!wXmMM0ES3_(h^S2?mnM*9niO#qEdbm_Nwny+4++VB!YDMge~SZ'
'uom#xC(s*(<ZnsNd@q<bfb9?x%A{v4C5@0wizldUet7h0FZ`jEKlRq5<B!qWSK&KsQxup)sW?nEo2O*<Lfx|lI-y?WG>t!#!7#nfJ6U+m`I2zr'
'yV_=xPxApEkHW2@kTRt~Cc$Y&HFqmI9}{jPhQWy~u8eCuQYPUkLyT)ME}`GPoYL`ld!PfKAxQ82FX*e$Ep~*wbO3yN=zNkLRj+pRnv%_-kc__0'
'#{)?sFGA`Tryqscy6)~u$!<EFc^(wdANjfN|D%;9ilG43aGg_*?WgHmhJwUd=|SgF?iFrYcla?S+&Rf`JZor>mXIi?w$%K6j6YkBGW`)eSP`Z*'
'|N8(ZeuFbq)`x{~`K@VjsW0RrJhBji1r$_PAKqv9U)LsBp&FzX{;8^Azfok?%fqYng&I}+wzU=v_PWyoopd+7Py=OtW#`+$^75*mS+U|+0U%%a'
'c}n`=DKh*3Qv87Yg4Y&n=4Hv2o7Q315N+gRfNdj_O_s!`&1e2Uj#d<Q-p`%Jp!oghSVbMQMzXlHZBDhL<IJh3iYzK({BE=&OPlX4^S*EPoWHi3'
'jiI_SAlciCfa8*o@W?{dHvjeD05zzX<z7dTf<fwtbU@{TJ1@tu<&rMlY?8+4Ga7zT*VZrW71uD^Kb%jH5BnfA^Iu`X>ppj&bav3M7=a^G59iBI'
'SC=9s8~nbobc?8`dV6eH;0md0{NN{pN(P?_lz>Ay@L~nt<1bK40|XQR000O8003fL+k%OQsRIB20|)>B6951JZDnL>VPAA|VQzC{a%o~^E^v8;'
'RcmkJI28Sk#DBQ!FO#f7W-7JP2q`O}lAR*818QMLD^yvA81QalN45iHXZzpx+6hfyA4Le_>zm_q?#qObspJE*Qx-PNSgr?$oiHWWa8UXWu49;3'
'1Cm*>=TPuM=_6=vD=iIJ1$M__`UFY>e-Sn&g!FqGtqP!YQ=u296hu)fZGlOtEaE6>`n`UyvHot9d|N2<_Gpf#K`NQu<zjV$nc>pC-|uC716DI>'
')>^E1z^Y8Bq2G@nXL4JyEgyoh8vcT>O7aMPj)3L4I#6De_Be!<QaL2>E6WWJVekv&!q{ad*7kYW?*V>1gE}zu6qts{_j>uDGM+m+&b@F8MC|Sb'
'^_Kf66K}cYavzX~bUYc++xy9ke!rVM&C*BuFde7&Bm!cEJd!X*=AsNjP`WWiEmFq!V$GBP2OOv0Cby|WT$j}aptE%F)~YBg%c(t<@1fpKA7+br'
'O6R}LJhqVb^98s!b=j(e%oWS1R?5ByKf1k3>3H(!us_<O{CHlM7f1b@M^{%_QM=QeWhN=uOK=qdk_+CUFk8_1-D5f$KRFUtBn-n2rmPBDH-|~C'
'8hcp(8NMyJv?c<IRp6^3?v!YYffDM14cl5xP7h94U9c<83Tnik91n60nNceWZY(QGlgxF+JGAx~-;KyI5bs3z+i2{o3}44!Vmok+!4wjH)HW{}'
'e~#R^KPAcQS*6HKxGtvnEj8AmstmRK#U{RwBCyz9Tr=1G$*1_!d4!w@gRxn*h(Cxk65p!9A$AL&+BVb30yS%kJ{znfi(Y>1Hn_|>W?Xes1*Ft!'
'#a1F0_SkXjSW)=~&MY8s;Nd)UDFBX={L^cTx<PhLrd`-hN%!_c#}e0K(YkzY-^SzqQz#C}3>tA~G-^Al>so8f1{z_Vy*|>pyH$7EwG1>9DC)72'
'u9#A-l@7?o@UKTacPJhb+PWi6CTo}az;+-POvu2`Qpa%k#x3VEz?mL1z1@eNn=82M=;qahAUNMn61ediPNwDcGyIgmhrz%r8hAw?!hdLbI--gq'
'#j7YFmGSexJlY*zPlEz)uS_u7wj+!d<H;1Ji#g6SxF3D}vKW0yVR?4Q^GUAn)(BK(aZt8Ri>3TV@@jb9B*R@oLG2pG2QA!W#SRJFZ5L-1MltA&'
'1@C=14cZ~lNn&yqrzE+-!312cRVL&%A(h<>K0Ctxg5BUHELq{MU<@ds)a8a!(nPJ!RKNE(P)h>@6aWAK2mk;8VqNA&PV&$S001N(000sI003=e'
'WNBeva%FRGb#h~6b1raswOQ>?+e{Mwp1)${`{MLTLPB|H51LA}4Fw4b7odG4ge=G2BwPG;-Zd}c_}_14cGs~T2RPlGB@}yKX6N-cGmGOmU(zB8'
'+2LWF@db-{oaT=#O_+@RC}PW0D!I=xp2tEl{*Xl?fS09MEc-c^X^{t<{Z3~Rx1^ckVP7PA*T8X{&OA?J=6Ul%6*>1jCgLp374wrMRlX8wB0C*B'
'CLfbPr1mpN6U86You5lPl$LM5Wt!M`xh#~3%CBOU<!QiWxmNKwn~R8--y-Ikugq7=NX#r#=kVFlGjm@mKNEK5=<@P>@(}P0Iqr14qvP*KY+Pk='
'@KwM6^mEsVc_@4jb>vwcNvFp`KKF}AjSu}u@@}WofdN)u<?+OyMLcx7tpAzK(lpxZFnDqf`b!ucJXmn$+U<9KBtl>D#%Zphvqv<D*pD>fnwXd~'
'GYxhKL90lSCwH!MI{9{d;2oTtoc;2yew-Yi9$!wrdDF?o+12#x<N{=H=E%I$9YpCp&w)*_BLj=O&Io5!UU2+<1cAHyG-BTulGnB9?hN0<!lRla'
'#c9ZykSr8Z^9N3)mM|cBe#djiN-Z=0KJm1gl`n7IAkEWWLzhN&&4n>l!)QU8J$hyUR9aa$$cx0)>XuwJNq4J8F@8G;a*l?m@MCC=&9CL&%cb;Y'
'j|$9`2{4gJ6j=y<HM1&zT<Pv!s3k-ePu#h)g|5Bz1iPNMz(G}kZpD)z4MnmTJH8BraJs+|lM1+rS5)wsr#Xv6!qHxB^Ts!D$pEP5O5)kB^D~}a'
'T<lftS$&KafO%K%1c(4-UXO0N>>V2ow|8h8C&M%M>|&DVY2IT${ixvdULl<M5VsYoL#wi2k*7uGWncmcNTTCI6<Nfu;p843f5ZM&adH5ksiuM2'
'_kpvV71CdD^7%+0oPl?ph@^O6nmW7NW}u?G5h2%_A@Vdu|3Q>+oh|t&x8`z7?{vE5s@Q)JaS_8pmo{uMM`P*$W*}}r?V+dr4V0XV%w6pT<(J#X'
'un#uWb#AnRQkLU|gg6<K{+lNTMV|9S)lXd=haMzhYbb)c?P{LVB#OJ5a^N;xFRSDYvmk{#>}fXp#3r3fDCUU750tp$9s)E1$I$6^yJSw+sG|EH'
'xKGVquh?u$%TqC3-F$5dCAaZJqN8_{UK^(TJZ2;I=@Z)>t_M(?L0>T{Iq{o#_&?kCtu|JW=_7BGrHz(Ks&6xNQ+FCAa{kXkK$)Q>@s=P02p1zW'
'3%-Y#*JJPCNRi-?UOXf01r3cp>^5>FAizNw3n?L0v0%xATQ=jMHh>Qn`8mQ!6r}-`F#9awkewJpDwHvdwj~P#G62R!La#h5fkkXi{CkZ6qECx^'
'!0S2`T8iZ&NSG^0%;Z{@pHBX9bv&JTU%yY!uFl(1J2=PFdRl*OSb<Xev`Db5A~QjrdLoTK?^$3qRi#!)k@$DMh)~>pka1p+vaBZhMpf)S&2OD1'
'nsjV%^bE0aDYZ!(R4KqPy5#$9jU;QbYnidInM{99ywfiz+-UnFx+Ep7osWNh*d>a_Sq#P9oedOEkAHX<mj}~J?`U%HZF+Wgs;Pp{xhsF`uG6Q4'
'i<0zh_*fY}*|S#)k7wwtmE9XGP*3|SbBIJ1^Ep)TkQJ;^j9XK}M?*M`&908y@e{GPwFOK(vs3ZKSYZ65*{@DCNf?R3@spLUy4pl-=*@Eu$5@50'
'mULJ2*2>;H5%Sd2uH#}ozeiwCaegb3kfI)x=FmcCJ<6y2HxJM$3t2!&Kbq(cITTeOuSl{&v3c&tTv}D>R%z=O6sFJ@VYEelkpM{iH}Rd$&}(%s'
'Gq_)h020=GPUB7);aF_bP_wkIS>NtM&a83?@~L2EzQr}ima!7L5<I+Yov8XPkGLjkVAlQ}#tFwm@TD(#1iJHdEs3S5bGJ8idMoS0l4W`ivO}Rm'
'Es&$vj_*YXMp|;vks?y*9DlH_GeLv4?B1+sNfoAr8drEcKR%z}+7KA3j(?l{{Bd=1(xXNX_q>PHL=dh!FHN)5sTO~rje$mgS+`%)A_`ZAFb+cA'
'1&KE132Igv)f&x!Zc#D-RzH-kIZzAK)j4vd!t0xwwavuKJYDGS$rQR0HX4Rh?gth`waS^+xv4D-_3PoywL|mHjVZpWGsaAYF=&SfCm^N_^d0=='
'9iJY2pLplTe@#v<9ONBikBV3k1N1nIgW_u&ew<%j+6@|L%gPIZcGSdKb1XQ?_VB~|-653^R#U_~`!r@lEZ~iZ_ES&3v%{b5NiXWOfndjE<tLQt'
'nLQE3=Z7av@tJBC9!h;>(#OjWiy!?=Jn(1@5X%rW*#Bf$dfvw%#1RF?Tk^&|3sPRkLQfS5pl6WAF|;<W`P~E3QeA@`Z*=!sxe6#}Y;R0EK=U|Z'
'36#nOI6=H~Y^zVjv{?lH(H$fJj!jgvL}~bM>~aaf&*qW8kSs1F1`iK-Xu?OLxahrk8X3*y)TTkLsNi*x``MBT0uD`k6<}A#BEBJgeHqbLODtds'
'a{8Kz*D?f5LI?R!PIcECW`P8AKUr`$f*#x|2lDi&r6C5ktY*Xw`aiLRqLjnz4>qDHD6QH%>gH0cm30#YLA2UpmyPyl(Y4bDFY>4?H5j_Pvnn>a'
'z1FknX;abYB|Vk%wSk*z=Wgw9D)Yi#*SzGW2C%6rhw{Ro@rd#_Ic){uzY;r8B%bSmGMQfxYB95pMd^ZED-`mUj38B-L4p1|o!MM2!1*X(m-q%{'
'TUk#IIz$ybx1=|{xX`xLRW0Qd$$YGyeB=nKQj8IwEeao8Mgb_WrJskU)FOHm@cXa^d^rJVz#{`6FJwd%u{4059e7jsoF|5=L=Xd52SjJ*k7}7F'
'w)7ZSx#<_<0sDi>9brAX+krx(LoMmlBWqJvMfNcz1YT$rAso6&3%`zzFU}`ZLVHjgmcxW5=GG4MBZFK(b%OzBnXIOX!^6|_$@l91@+@2XW2X4~'
'ZZlP^xWR1k=Wa_?|G)1O=$g)6H4iR1UL5=R;~{*xbR!3j-FtBXlrfcyFynHys_dMu0ZY4?l86YKG6?=^_dB_wdjt%uL7akZ(?kSrwX5=>Fo6}d'
'8;G@Q6sSj|G7JV;8b#P1HoMFU2sRG{xU}4SF2j7ob)w-2+7|LKn_gk-liB$!Ee)rY#tewuwz0MD+_j@g-__e$E3gQ=tvj%MgiW7UPT5ZRHNjRk'
'CX6wm<US#B=!1-(HJhQgiH*<e*9R?<m76dy?FS@$5V!n-%u_b3n?-MJTQ};*-JGZxt7s3wJOkw!B(GmJ@Q0-jbq1DLn-!+%l?`rH#+E*}D)i=V'
'cb)#r-LEdL(CNn3+;67rXF_kR_^ie-@K_fhs<nr1KJ5;GPrE}=Y|DuA%pDF!YuBFIj_b}~gGu4P5)<g%TRd+Kc$%B<eWBd4uUjWzZVxM%d8rpq'
'9x`l$U#5O55ZGR4puVJDmA$fI4x&d<LrMy^w_)omtZq0J6flsS-@3!q7^Dg_)??c(!{$yGzZEB@pyb_s|8b{w#lL=a&@9eY`Kr8JC|m6wafT(N'
'+to+u_u9Wi+8nB-V>9^i2XprECXL!B0|s-kfVibn&<}rO`+<|#Phdw*EhBbTYyY~?;&uKFP)h>@6aWAK2mk;8VqN}DK$DsV003$l000sI003=e'
'WNBevYHwm+b98TVWiD`e)mdF{+cp$^_pf02605Tvx5$9On}(&W*QIOfz;S>LK_JLB6|0d&jiTBZp8xwUKP-tNmAGwCV0wsb^77-JbMED(6EV6*'
'K@b(ZU?d1oyv``&C`?nzLmtyK?{st<&G(;pye6GUjO3dvPH*?4PSZ_k&B#9m$vF(~oKN12kq_esw}g}QQx~6)-<+KW?<TK;%h_Z)4ul2nA<W2{'
'a)P}<&RLvwy+J}h6V~-QuO}a7)5-hb;{EA|aq#N&avWSu-<y-VZN#*-37PtUkFnQj?&9>~EckQ$_rqJ&M&`7g+1dGcay1Js$FPexmoUXTyzjmo'
'40|X_Xvn({%-9$!X8s9#p4XX<XVbIsezF%mw2ss6Ne{*8gY)TOVxE^`9uFv3Bw?PTbFz%XcXWQqDI=o}f@eu06eM&JCP6M*yE#cBPxXmMvK*%o'
'3kMltd7Q(V3$iC%4nbj#YBQ8W839eDMpw0@VmesRG>UK4qNx8X;zgE_Ye2OJbF-UjA8~|0k;u()WhU1R<D8&rkqViP8KbOC8z|1@tZEk!083Cs'
'W5pK!A78RfT@S6PA#?Ax*R~e}RvC5Oaz=q{ul%+qd_|Wdfo;{8g?!a$dR~3}nA0?v)8%G_mhplsZXnQa(7)&dO$n$|ys80V0zn&`3{OUAPH6)D'
'zlTY#4J66?)G(NmJcA5F^i&n5^AHB0JF<b%!h=CWFBM`16Dve`D~9a0!j~N`dsie}f@;BHYw(5t>Psy9fMs7q^GGb#Gi^>63zBi@!eN#r@j|A<'
'V}S>*EO&LilvwRaHodCt=o-JKDTEU5&o&vsH?Tu{l}<SlqL&9m=xT(<_hMUMWD-zCEM+L<oUAhrfeX@1Z%J3@1kc<IXB)FYJZw*jUlC==s{SZZ'
'^_+b`kGn(T;On>BNTA@Cpt@~D#L<HLjVx*#dsYiDR*Dr)yk+S+r1g*&IjC(3F!)G_*}b?6!2Po%Jd6_wqn=i{0%s5;?k-ln*xr?Rh~&7U`hZ0Y'
'4K%#eMxcf*zDn;>`Z)#G;4DBN>0F!>`$k~7AhL1SzNAi4tqVGmVe4G@e7_BSmXLIrQ8?iADr5=7pduyr8Ch`nCi4nzoevU0i8d$q{Cf0aI2_*W'
'+#{TlaEjm;^y6^ohNQHYeg_bKfg0I$bF1Oo53_8lSO-=gM=`P>ERp+THoHJump{8U3=hZ8bWRJlmf7WpDpx^`05&#)hPPnRF)T!c>QwdvdI2ox'
'JZ8@$P#WY3Az61gI6==*w>D6eKqdUPI$aRT?V||7hQN8bt+u(tUUlAP1rK0U+i?qBLEJoU@8Y!Cq?oae<qcGhU<!w=b;trqMZ(>E`ka8tTt~D#'
'fz5LXP*x>6973{M!zRA!1g;l}-oWzQ^x00o!fm@%d2Y9B=Y9RQ4HN*rdfD9)17H3%J5AZDP_ck`=9gEIN9#Om=?JoEUSy&^h8T3+bo{OO`;6^='
'MrB?*lJFiOh=U`w^bu}s+z{EoF7?ENFRo_T^f<!DkWUFq!{k^d{XS5XA8!>=yM}dn+WK3|Gb}h1rIM3MthmIZR#LHM6HHop#fnidX`>cPR>8o*'
'?ymY;i&8jmrB-Ed)`L>Ug0reUJUDbZcg4tlUA3b3f0J3Q5PV2+Z^z%JxUvNEXRIJ~0&94cdx65*E~;G>i7dmd+N++Lp;s8Ga!k}b);&xZ!Qycq'
'r=V%L_X;X|QZXJPaH&yCmO_)|Q$SMrw&m8V<^C$YoBDG|a^wj7n895rp~W&vLinf_^d6@UUra7%qT+KhB&Rvn`^WBj+jwf(0xL%$F7*jurVB^#'
'?<~9<WL-;EZkqTCzP>7yp8!57_BMel_$rw<q<x^fhTL04wn)Nwtzm%5oiHt{ReZY&7GW076JlP#zY{_S2u$w%OXVe<1o6l>%wKmDn(JR56;?gu'
'8@R%Kl<_$U9TGEv=JbxFmGlJCUxA~}Qm*J}IxH82@AAES8YW6_!L9;Z{IMVu4tAY5hDrR1V9$awiqkk>9gbs*AnL_&S@$}U=67b`fC9)WW}b*O'
'1?;y_re+X7JzA`V3k4Zm{^atH9Ag!gl7#Bg@w8N!M~Okq6M?Gz?(hBnvi#)4-wC}Ek%s*@p_F)`enBgVcmS*R%iEU}lls1^e(lkV<?b3%O(}mJ'
'<12z*mpc+aU~@*3ewKuA#mpGJ-w3xl>IUtb|7-x3{{v7<0Rj{N6aWAK2mk;8VqLIRB-*_W0049{000dD0000000000005+c00000VQF7uZ*6dI'
'b7d}Yc~DCM0u%rg000080001DT}C6)R}}>S09gwF02KfL00000000000HlHF4gdgQX<uV+ZE$aMWnW`&ZEbIEb1rasP)h*<6aW+e000O8003fL'
'ErMe`4kQ2oWPJbt6aWAK0000000000q=7IL003cWUt@1=aBp*EUu9%zbZ>GlaCuNm0Rj{N6aWAK2mk;8VqMrKq6xid003u@0RR*L0000000000'
'005+ca5w+}VQF7uZ*6dIb7fy;Zf9w3WiD`eP)h*<6aW+e000O8003fLVimZp?G69{4mbb+5C8xG0000000000q=8?d003cWUt@1=aBp*EUukkK'
'aCuNm0Rj{N6aWAK2mk;8VqFwY6dT+N007G$000yK0000000000005+cdawWhVQF7uZ*6dIb7fy^VP|J-WiD`eP)h*<6aW+e000O8003fLzIox^'
'vjPABKLr2)7XSbN0000000000q=AUM003cWUt@1=aBp*EUu<t<VQgPvaBwbgc~DCM0u%rg000080001DUF@E59xem`0O1J$02u%P0000000000'
'0HlF%zW@MXX<uV+ZE$aMWnXM>V_|Gxb98TVWiD`eP)h*<6aW+e000O8003fLF^PUL!wdib6(Rrt6aWAK0000000000q=D7L003cWUt@1=aBp*E'
'Uu|!6X>V>WaCuNm0Rj{N6aWAK2mk;8VqFhu;sw?V006xt000yK0000000000005+c%FqA+VQF7uZ*6dIb7fy}Z)|B}c`k5yP)h*<6aW+e000O8'
'003fLV0@xi2MPcH@Ff5M6#xJL0000000000q=C}h003cWUt@1=aBp*EUvYI|Y-x0PE^v8JO928D02BZK00;m80AgK<%;2L?4gdhPH2?q@00000'
'000000001_fehyW0AXogV{dJ6Z*yf|a%E;|Ze?v{Zgehic~DCM0u%rg000080001DU3FDS8CeYg0A(=%02BZK00000000000HlGE^#A~2X<uV+'
'ZE$aMWnXh+Xk~3-E^v8JO928D02BZK00;m80AgJt;+d5m2><}v8UO$m00000000000001_fgJ+@0AXogV{dJ6Z*yf|b7f<6X>V?GE^v8JO928D'
'02BZK00;m80AgJbyOhHo0{{SU1^@sP00000000000001_fo}`}0AXogV{dJ6Z*yf|b8mHSWG--dP)h*<6aW+e000O8003fLJreJsJ_G;&<p=-('
'6aWAK0000000000q=CK<0RUlXUt@1=aBp*EUv_D8X>V>WaCuNm0Rj{N6aWAK2mk;8VqM#UiHE5J0009B000vJ0000000000005+cE))R(ZDnL>'
'VPAA|VQzC{a%o~^E^v8JO928D02BZK00;m80AgL{Mo#k33IG5k9{>On00000000000001_fe07@0BvPtX<=V-Wpi(Ja${w4E^v8JO928D02BZK'
'00;m80AgMKPC%2I1^@tR8UO$i00000000000001_fe0f30BvPtX<=V#Z(?6_bZ>HHE^v8JO9ci10000J022V^1ONciDFFZg00'
)
BUNDLE_SHA256 = '38d6049ccdd1d9063b6a335321b3007b9f92c9b8393ad651de521a1874a71732'
raw = base64.b85decode(BUNDLE)
assert hashlib.sha256(raw).hexdigest() == BUNDLE_SHA256, 'Damaged code bundle'
RUNTIME = Path('/kaggle/working') / ('sh01-onecell-' + BUNDLE_SHA256[:12])
CODE = RUNTIME / 'code'
CODE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        target = CODE / name
        if not target.resolve().is_relative_to(CODE.resolve()) or not name.endswith('.py'):
            raise RuntimeError('Unexpected file in source bundle')
        data = archive.read(name)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Bundled code was locally edited: {target}; preserve it before rerunning')
        target.write_bytes(data)
source = Path(SOURCE_PATH).expanduser()
if not source.is_file():
    raise FileNotFoundError(f'Mount the video and set SOURCE_PATH to the exact filename: {source}')
if shutil.which('nvidia-smi') is None:
    raise RuntimeError('Enable a Kaggle GPU accelerator before running this cell')
subprocess.run(['nvidia-smi'], check=True)

"""Repair a private editor venv while retaining Kaggle's CUDA package paths.

Uses only the standard library until pip is working. This source is inlined by
build_one_cell.py so the notebook does not need a checkout or a second download.
"""
import importlib.util
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import venv


def prepare_editor_environment(runtime, *, host_paths=None):
    """Create or repair the owned venv; never install into the host interpreter.

    Debian/Kaggle can expose dist-packages in the kernel but omit those paths
    from a venv. A local .pth file adds the actual host paths after local packages.
    The host's pip can install into a pip-less venv using its --python option.
    """
    runtime = Path(runtime).expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    root = runtime / 'venv'
    python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    log_path = runtime / 'setup.log'
    child_env = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        child_env.pop(key, None)

    def execute(args, *, check=True):
        """Keep subprocess startup diagnostics, including sitecustomize errors."""
        result = subprocess.run([str(x) for x in args], env=child_env,
                                capture_output=True, text=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(
                f'Editor environment setup failed. Log: {log_path}\n'
                + (result.stdout + result.stderr)[-3000:])
        return result

    if not python.is_file() or not (root / 'pyvenv.cfg').is_file():
        # Do not rely on distro ensurepip, and do not clear an existing runtime.
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(root)

    probe = execute([python, '-I', '-c',
        'import sys,sysconfig,site,json; print(json.dumps({'
        '"prefix":sys.prefix,"version":list(sys.version_info[:2]),'
        '"sites":site.getsitepackages(),'
        '"purelib":sysconfig.get_path("purelib"),'
        '"platlib":sysconfig.get_path("platlib")}))'])
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    if Path(info['prefix']).resolve() != root.resolve():
        raise RuntimeError(f'Refusing to install outside the editor venv: {info["prefix"]}')
    if info['version'] != list(sys.version_info[:2]):
        raise RuntimeError('The editor venv uses a different Python version. Use a new runtime directory.')

    local_sites = []
    for value in [info['purelib'], info['platlib'], *info['sites']]:
        path = Path(value).resolve()
        if path.is_relative_to(root) and path not in local_sites:
            path.mkdir(parents=True, exist_ok=True)
            local_sites.append(path)
    if not local_sites:
        raise RuntimeError('No package directory inside the editor venv was found.')

    if host_paths is None:
        host_paths = [*site.getsitepackages(), *sys.path]
    shared = []
    for value in host_paths:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if (path.name not in ('site-packages', 'dist-packages') or not path.is_dir()
                or path.is_relative_to(root) or path in shared):
            continue
        if '\n' in str(path) or '\r' in str(path):
            raise RuntimeError('A package directory contains an invalid newline.')
        shared.append(path)
    # Also bridge distro purelib/site differences inside the venv itself.
    for local in local_sites:
        paths = [p for p in local_sites if p != local] + shared
        (local / 'sh01-kaggle-packages.pth').write_text(
            ''.join(str(p) + '\n' for p in paths), encoding='utf-8')

    health_code = (
        'import json,sys,pip,wrapt; '
        'print(json.dumps({"prefix":sys.prefix,"pip":pip.__file__,"wrapt":wrapt.__file__}))'
    )

    def healthy(result):
        if result.returncode or 'Error in sitecustomize' in result.stderr:
            return False
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
            return (Path(data['prefix']).resolve() == root and
                    all(Path(data[key]).resolve().is_relative_to(root) for key in ('pip', 'wrapt')))
        except (ValueError, IndexError, KeyError):
            return False

    health = execute([python, '-I', '-c', health_code], check=False)
    if not healthy(health):
        (runtime / 'dependencies-ready').unlink(missing_ok=True)
        spec = importlib.util.find_spec('pip')
        if spec is None or not spec.origin:
            raise RuntimeError('The notebook Python has no pip. Install pip for that interpreter first.')
        # Executing pip's package directory also handles host sys.path differences.
        # --ignore-installed makes both bootstrap packages local without removing
        # any globally installed version. --no-deps prevents unrelated changes.
        print('Repairing editor pip and startup dependency…', flush=True)
        execute([sys.executable, str(Path(spec.origin).parent), '--python', python,
                 'install', '--disable-pip-version-check', '--no-input',
                 '--ignore-installed', '--no-deps', 'pip>=23,<27', 'wrapt>=1.16,<3'])
        health = execute([python, '-I', '-c', health_code], check=False)
        if not healthy(health):
            raise RuntimeError(f'Editor startup check still fails; see {log_path}\n'
                               + (health.stdout + health.stderr)[-3000:])
    return python, child_env

PYTHON, setup_environment = prepare_editor_environment(RUNTIME)
# Use a private environment; preserve Kaggle's CUDA-enabled Torch installation.
import importlib.metadata
constraints = RUNTIME / 'constraints.txt'
constraints.write_text('torch==' + importlib.metadata.version('torch') + '\n')
setup_log = RUNTIME / 'setup.log'

def run_setup(command, *, env=None):
    """Record setup failures without putting secrets or huge pip logs in output."""
    with setup_log.open('a') as stream:
        result = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT,
                                env=setup_environment if env is None else env)
    if result.returncode:
        print(setup_log.read_text(errors='replace')[-4500:])
        raise RuntimeError(f'Setup failed; full log: {setup_log}')

stamp = RUNTIME / 'dependencies-ready'
setup_version = 'bootstrap-v2;torch=' + importlib.metadata.version('torch')
if not stamp.exists() or stamp.read_text() != setup_version:
    print('Installing isolated editor dependencies…', flush=True)
    run_setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-q','-c',constraints,
        'fastapi>=0.115,<1','uvicorn>=0.30,<1','pydantic>=2.9,<3','requests>=2.32,<3',
        'httpx>=0.27,<1','python-dotenv>=1,<2','Pillow>=10','numpy>=1.26',
        'transformers>=4.51,<5','accelerate>=1,<2','bitsandbytes>=0.45,<1',
        'json-repair>=0.30,<1','faster-whisper>=1.1,<2','sentencepiece'])
check = subprocess.check_output([str(PYTHON),'-c',
    'import torch, bitsandbytes, transformers, json, pathlib, site; '
    'assert torch.cuda.is_available(), "CUDA unavailable in fresh subprocess"; '
    'import sys; roots=[pathlib.Path(p) for p in sys.path if p]; '
    'libs=[str(p) for root in roots for p in root.glob("nvidia/*/lib")]; '
    'print(json.dumps({"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"libraries":libs}))'],
    text=True,env=setup_environment)
gpu = json.loads(check.strip().splitlines()[-1])
stamp.write_text(setup_version)
print('GPU ready:', gpu['gpu'], '| Torch', gpu['torch'], flush=True)
if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
    run_setup(['apt-get','update','-qq'])
    run_setup(['apt-get','install','-y','-qq','ffmpeg','fonts-dejavu-core'])

MODELS = Path('/kaggle/working/sh01-editor-models'); MODELS.mkdir(exist_ok=True)
def download(url, path):
    """Atomic public model/binary download; retry transient network errors."""
    if path.is_file() and path.stat().st_size:
        return
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'SH01LocalEditor/2.0'})
            with urllib.request.urlopen(request,timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response,out,1024*1024)
            partial.replace(path);return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt==2: raise
            time.sleep(2**attempt)

# The standalone Piper distribution avoids Python-version wheel mismatches.
PIPER = MODELS / 'piper' / 'piper'
if NARRATION != 'off' or 'voiceover' in REQUIRED_FEATURES:
    if not PIPER.is_file():
        print('Installing local narration engine…',flush=True)
        packed=MODELS/'piper_amd64.tar.gz'
        download('https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz',packed)
        with tarfile.open(packed) as archive:
            archive.extractall(MODELS,filter='data')
        PIPER.chmod(0o755)
    locale, speaker, quality = VOICE_NAME.split('-')
    language = locale.split('_')[0]
    voice_url=f'https://huggingface.co/rhasspy/piper-voices/resolve/main/{language}/{locale}/{speaker}/{quality}/'
    voice=MODELS/(VOICE_NAME+'.onnx')
    for name in (VOICE_NAME+'.onnx',VOICE_NAME+'.onnx.json'):
        download(voice_url+name,MODELS/name)
    download(voice_url+'MODEL_CARD',MODELS/(VOICE_NAME+'-MODEL_CARD.txt'))
    run_setup([PIPER,'--help'])
else:
    voice=MODELS/(VOICE_NAME+'.onnx')

WORK = Path('/kaggle/working/ai-compose-onecell-jobs') / time.strftime('%Y%m%d-%H%M%S')
WORK.mkdir(parents=True,exist_ok=True)
environment = {
    'PATH':str(PIPER.parent)+os.pathsep+os.environ.get('PATH',''),
    'HF_HOME':str(MODELS/'huggingface'),
    'AI_COMPOSE_HF_MODEL':TEXT_MODEL,'AI_COMPOSE_HF_LOAD_4BIT':'true',
    'AI_COMPOSE_HF_MAX_NEW_TOKENS':'3000','AI_COMPOSE_VISION_HF_MODEL':VISION_MODEL,
    'MEDIA_CLIP_WHISPER_MODEL':'small','MEDIA_CLIP_WHISPER_DEVICE':ASR_DEVICE,
    'MEDIA_CLIP_WHISPER_COMPUTE_TYPE':'float16' if ASR_DEVICE=='cuda' else 'int8',
    'MEDIA_CLIP_WHISPER_CPU_THREADS':'4','MEDIA_CLIP_WHISPER_CACHE_DIR':str(MODELS/'whisper'),
    'AI_COMPOSE_TTS_MODEL':str(voice),'AI_COMPOSE_MUSIC_PATH':MUSIC_PATH,
    'LD_LIBRARY_PATH':os.pathsep.join(gpu['libraries']+[os.environ.get('LD_LIBRARY_PATH','')]),
    'AI_COMPOSE_USE_NVENC':'false','AI_COMPOSE_VIDEO_CODEC':'libx264',
    'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
}
payload = {
    'brief':BRIEF,'preset':'viral_short','target_seconds':TARGET_SECONDS,
    'output_width':OUTPUT_SIZE[0],'output_height':OUTPUT_SIZE[1], 'output_count':1,
    'captions_enabled':True,'fetch_assets':True,'llm_required':True,
    'refinement_iterations':REFINEMENT_ROUNDS,'local_asset_library':LOCAL_ASSET_LIBRARY or None,
    'asset_sources':{'local_library':True,'wikimedia':True,'pexels':True,'pixabay':True},
    'editorial_policy':{'source_audio':'preserve','supplementary_visuals':'require',
        'camera_motion':'dynamic','motion_graphics':'hero'},
    'editor':{'enabled':True,'narration':NARRATION,'narration_budget_seconds':12,
        'caption_style':'highlight','theme':'editorial','progress_bar':True,
        'music':'library' if MUSIC_PATH else 'procedural','procedural_sfx':True,
        'transition':'dissolve','min_graphics':2,'min_visual_layers':1,
        'required_features':REQUIRED_FEATURES,'visual_review_required':True,
        'strict_requirements':True,'publish_outputs':False},
    'sfx':{'mode':'adaptive','volume':0.20}, 'music_volume':0.12,
}
config={'source_path':str(source),'env_file':PRIVATE_ENV_FILE,'environment':environment,
    'work_dir':str(WORK),'payload':payload,'timeout_seconds':TIMEOUT_SECONDS,'keep_api':KEEP_LOCAL_API}
config_path=WORK/'private-config.json';config_path.write_text(json.dumps(config,indent=2));config_path.chmod(0o600)
# Run outside the notebook kernel: no stale imports and no foreground Uvicorn cell.
from IPython.display import Audio, FileLink, Video, display
import numpy as np

def beep():
    """Notify at readiness, completion, and failure; browser autoplay rules apply."""
    t=np.linspace(0,.5,4000,endpoint=False)
    display(Audio(np.sin(2*np.pi*440*t),rate=8000,autoplay=True))

beep()
print('Running the integrated editor. Evidence, drafts, and logs are saved under:',WORK,flush=True)
runner_log=WORK/'driver.log'
with runner_log.open('w') as stream:
    process=subprocess.Popen([str(PYTHON),'-u',str(CODE/'ai_compose_kaggle.py'),str(config_path)],
        cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,env=setup_environment)
    offset=0
    try:
        while process.poll() is None:
            time.sleep(2)
            with runner_log.open() as current:
                current.seek(offset);message=current.read();offset=current.tell()
            if message: print(message,end='',flush=True)
    except KeyboardInterrupt:
        process.send_signal(2)
        try: process.wait(timeout=25)
        except subprocess.TimeoutExpired: process.kill();process.wait()
        beep()
        raise
with runner_log.open() as current:
    current.seek(offset);print(current.read(),end='')
beep()
if process.returncode:
    display(FileLink(str(runner_log)))
    display(FileLink(str(WORK/'server.log')))
    for diagnostic in WORK.glob('*-output.zip'): display(FileLink(str(diagnostic)))
    raise RuntimeError('No passing final render was produced. Review the saved diagnostics; requirements were not silently disabled.')
result=json.loads((WORK/'latest-output.json').read_text())
for output in result['outputs']:
    display(Video(output['local_path'],embed=True,width=360))
    display(FileLink(output['local_path']))
display(FileLink(result['archive']))
display(FileLink(result['receipt']))
if 'api_connection_file' in result:
    print('Loopback API remains available. Connection details:',result['api_connection_file'])
print('Save a Kaggle notebook version with outputs to retain these files after the session ends.')
