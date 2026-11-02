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
'90;gbcR~ICG+4NFPrdTZHVGY>BBvhK&uvuhdw=ggP)h>@6aWAK2mk;8VqLP&f)Q^90090D000#L003cWUt@1=aBp*EUt@1=ZEtRKE^v8`SN~7k'
'HWdFIiT~k=O>#1h+`-sXT}vy<Dz$3Wq0(-e5Iy-4-^FVZd$z-s0ss5nvz<%s=tL1p@$>us<@4OVy}h^M=3rrjfIsCIS&Kq)I8$}48Vg1^BgHkZ'
'U?p7%+Nf*M@Fo|)n*zjFUCCTJz^`bmu-Wd;?#_kaCNH5Dju+f<kQQ>}mUs{#u4N%megutjq)#mXe-}J=Em|?6LTh-R!N*cSrE*^F?p#Y7F0Z9n'
'2?N&2sykY6$e62ID=P|E7+#|v@LpUaBX^FYE5u=gSO2ua-0bc+HSnnrHaAi`NfG0XY!(&K*Uo^=mGRN-?d|REER3pQHe0l=HDWdciMEZyj9^II'
'YoRnmYU%1~eJ~<Qx2CG(Jk#7*(a9stpGkOkrw|L!6_-td*>6QU-Q5BF!Gg4=@mjOCLC3iki4mD?=ZRs@E<ZoN+JDYQzyLqinJAs686Kog(q6l-'
'^sG>;rc%6^q4@+OS0nf$o1(La0iPSOpj<+0`MeUC&9<I5Tvj7cm|3%u7=tr6KFug9w7^f}svF2lk$=IeC?l7$;g#<XQhN$o>6Wrw1bt|o8ikso'
'5xKCI8;J^d13b@#cHVO5E1&_bWaF}+?X3{nTlXOyl*&@7Dd3#S@f1~3H#pN;DwFpbJXbBS$Tu(-Id3WRq>U7P`tT08*4Q)Xplsmu{LQ+dGdMTO'
'sa#bDD2xTaj)qCfVwMEF{`m3yB3oxL{E?Rm$Hc-3TvDF1rEm%Bf@kRnda5T}fn4}nz{v?PS@We}ASzUUxy>02ER+FXZt^%<i)ExA)JEx*RyNjA'
'TZi88ZIWV%{H-T-y)qE?0jS2wrWHLkSUcI2+OXK&aN{gNNW#Wrwyn5R!%FA0oif-Dq@D}gYfHwD3Z`Z~HnGz;xp-Bnm1j$WKxVNgYvEJstuc>e'
'@sfI?;qkiF)N9$Q1I0|4CEHX#wA5?C4lrH|HuQVdh~b^))~=K(y1Vbu=g|R6nvlQ<j*ov%2Vsh@G#~H>MgPheWge{>mL=w8i#3AFAor0HJYB;Q'
'xiwqkr^VFXP^vbO0|&6JoNxC*e`@SviY1DKCib9=cy)W91_E)}Nbug+jb^0w4<%iSAmFEi$jo>0<?a1ddO5l3MQC_Yq#qpd)pY|M4E4NtkE|81'
'=LH8U<TN5N33zED4&3!F66msFVd7Gpg@yvp`7Y2dN-XJnq_YA45Ab_~ShV1^tRy#>K!TEPwT{9DUa#E2D#jwAci1QoW*rwp&0SvNYWSR-olWk}'
'&fw)=$&2a1#cBEi|Ao6B)4TI_jye%tYgChB97E4Xq~y)xP*6)BWuIUogq&=+wFWeL2&gc9Ue;q%1GnK~yb|-8#KTD~m;IsE$L#)lx_lcvAKLO4'
'<G;1=>cdGlg9Eyuzt317xt0G?<dYPh!;{HRhm)gY|Dn#Tv5k7pl`LH8S%>If3b`zuC#UcXj*cgf+Nbq-^2kPi&>eXHsNy)q87;%6NNyS8mvA##'
'uqn&KBXP#|>7hZxc<Cn>FRw5lW>sr)fsAViWur}d%t==vM;NbuoC7%w6n+N}G&c1NZ>TuG4an)X=3y4t6hCW+0;Ny)Kzm%oEDivhZcM_Kh3?L)'
'8&3D9s0Cl2M$X%T+o#4!>sY$ao&wwVq%(YBNcYfxDjA#77d1J)v{A(wBJL93Q;3gg2U-!OW376i4sYJX!)s^)n(bP;lPBFkuOFt!WJxv_U1MQs'
'p1X_KnSMWy$FcI8yK%f4b1gGn>S38n!3$yR$u0YYYw_T8i7!KJiHp~h!*}$$e}&O2Gd{{DkaX&oh;;2dLo4H#++eIPF+!)Bu`MTu2MdnR2`vv$'
'9lq*Pc0aI&tBKcSjOQWk-ELvHd=Lte@V7<!h@l_Ca5Rmikf-hD^krCD6K|(p(>{ZRtc1JK!oO3J4#&O$ZT_vs=F-(%KPYggojS{EPIL0>hzPzw'
'Y)U|JG#U9*I${X-8a?gz+_%(88QHKU?+EDb2#W4DRvj|oZ2+F>;n9=H{m3DwYTZe4_{->clBSzYkQt5-{2AtNhIx0b?Ou3sh5IIntd!gC8WX|<'
'y+_z!-w*J{tA!73L+1U%1)Dqmw=<-C3LUa|*~_TzD)eHm3WNyOHeVVXbqIH9x;Ya6`&5aD_#aS90|XQR000O8003fLErMe`4kQ2oWPJbt6aWAK'
'VQF7uZ*6dIb7fy;WNCD7axQRr%{^(8+cuWpQ<eXKX>S!JH=BLLPF!;6UB_`I-brGw+sV$BM#Z8iNMf22H9Y#5-v00V4gdsrsN2VfRaIh%1Rfq9'
'-i-(J!NI`~dA(i8g^24a-^5j#2)RhByqE}inJ(luk>cou*v3TxFE8oGLYC=ji{Ik1lvRn}3TQ2_g0H{&`m5hdS&C{cg}hG7D&4Nc20+F_ZWp^e'
'-B!o{tYs~I@$&q=xXOzQS%`SM5J|q-<s}Xsh}W00xDksYURGa!wT?@X!YKLGRxHv&CeY;OxroIwPBXw9hc456E0TCCwi3FE9R7Dzq?IH*uF~xm'
'(MakF5Ii_I`1-45k#9s4E$gZ-WE6=M#x5$rW1Ckn&wN|rO!U+8W}BqB`8|(I`Rx<)^UpHhnjd*-z82DaE!TCGW@^}7T&=TouE%@_KXEX67pD=^'
'K}Kb=mYZ0&dWn>NFM)Jr<%=HxgLAps$SwSMr3eRDzsusSFaBD@yLFoIkMrd<P;*;#2iWCVn%oG3ZJB46GUC-w<4nEp_Dc)h2cN(aP@iZA!*v_~'
'vyQX0GNYv1at908^zhB20{BA~T?m^TXQlyc;#~wxv!M0)`m2RpiZZ^GBP6OXF5|40<H^@w3HZQ5zmr7?JlL>u3{0LTahBbPGKZlnSrqk-2FN79'
'VaCnkk|&o*V4)>UF(9r%sMTgCLR!P%Dy`O0SudCA^~ek0g*P^B0%E_YlGoJ;86;TLn_W2~1aZ!QP8FW{AQUBXU|c3?8vYb#r5xkKBwql>hF)DQ'
'kDl3J3kg(9c)sI?21&e&=P6J+mF1{lHqmq_Bry3_-10B{cv%iKh*)~PiBKH^dJvU)T_my$xDjw_5rZ(Ih|cpo)6fG8hG~?8=#{|=sJ6Wvc`x5Y'
'uipOp?(O;O=;itO>krW{Z+>|H^8NpK<FRicvy;~{8%Bd)H{)9aynFrk*T0-cFW<e1{`R^9HocDL@y#$?e%}xH&EH@C@bZ5KAlJaLz#{8G(7Hh{'
'?B9PofA`DF^Pi)auU@@AKktEmm0qNzG6x4REu0A#c_J5e0jp9judTi;(t-Ql?pYBR5TdN9f-BG;>k-g!Ckt;(^Dqm4nWu?F4a?t#7yZMB^XS*N'
'KfeA2=I2asQ<rI?nMRMS!Sl<y$c6#``uq8tSJAtdAAW984h#Y4n~05(2HX$hk2mLU-u@Q-{F6ozhuh!w_xHajS{PFBc$<77GlR*UrJ6v>QSX!^'
'(pi>f`T*-!R`F&B1`PC{Ox9v9<EnfvcJ(%?YBEB{t9Vz!V4FDI8Xe{<8<%8px+xpQ1PZ*)z#JrX0jjhDB{iM-@E^Zj!t4+SP(7geib^gRC0|^~'
'n=p$v^F=JKCgN)90a``n%`~$3NtXsJu>y8A4aJduIXya^O+C=#@bHBQzI}=`&E$5Z-jBtLP@FxnAvc%@@oooQP21<rOqw3*scw#_j2%o18I&?E'
'fPV{b8vNzoXaDB;qO1yPipx9l@SEp*z&!!HNOz<0r{mPn#sbmb-O<O~q~Voix=qV98AUA%zF-r{qAuDz^a|XRszBZ#1|-J~7PQ>PJM?wHmgSna'
'aMi$$Wnxt*e@{7a35?V5P}}wv60nl!W@!5xZS!ISCLR3;GWdPn<aPz!Oc&rVWwI5$;_c0-3@*S5iob-yW3A>9kCw=w2suD6D=OK^DedKf8JAN0'
'ftK`jQRKzQ`%!s?0(1vBTQY-Rf}w(#q&tu*<-mfSFS8s-6BknKz?j0nprUWc?6U`g3eR9_j9s>g@Oz5Qk(l5os%h{zFz1DMVcL^u0QO0~U6Ac^'
'xy$sTcH(=nNzI}M0Lk>kIpN|@JyhI?zc=$cPsU$>bWAmgGeb8jU1$Q5lLSwm!{nBro57JT!6O2eSb(N1HpsRZHUJzn3OAa1hzW>Ei(y>W4v2Aq'
'Z+17Mg_OJSLs6px<S=|^F_b-PmK2#NUoJgsrUczss*>bYh(|cPmE20`*eG6rfXhG67r+NaoLn%NN8&e9w79aHba53Umw@=JP@Hu1vIxYFIdQ{A'
'7AtvNNO%UF5TFT~f<_0cRq0<U3;(3ZOQLlM>K=T^(G{z3>_}umf;7q+SV#e}`SE*fLFyQ2`dhx0uA($!Q23>OG$3EJK{B%*ir}oJEPI9yy8yY5'
'=2~b`)*DBDj#^@|kEt2h1{Ew0X2Jq()dstemBqo0X=WgrYdI(i2jiiJf%}=1Tj`0!^&-th7@~J3OOPXJGwH`LgiU9H8m&<ChvGCiX%EgJQ?%kZ'
'u<VnMN8%+|gC#^lb6|LZ5-8GA{IkwcFi4>rjlg1}{SJ(~(Lu?8L=@x%4OCPHn|ed!b7U1>7<8KEW`vNJWjdK`;_K0=Pfi0p8Xq64M+)|MJey9`'
'_nAvdjH<4E0)^z1+j>KSKGNf~jXY2c2O_#_?8rw^@LSD^eF_<0xQ{W$gBf_j3;xVOkCUr0WglZ^9E?VM5x?o=^lUbE;lu#sBtWESBFHr-0zw~%'
'K(joqh#u3ES>vF&-RFYv9%%vd0pOuP;++LgnWBj4`1thMc-)}_O@hLeqZoa+wuqB#*mTmqjg@Q|p+yh9b)H{%ebexWyP_mU=c@~zI0B~@#ux%G'
'!nsdqh4?>`=!9I6(5v!Y{{UHjC5tGN%W7zV7#5bu6M7{WIT<bTt8G899gKpG<4q<*FHRB|dI%~oMTsC;d?L+n5laq0(at-1RAo3_Vz363+)Qrc'
'NpQBjJG!0Izp$!fTS$K_@*Vn7Al7LGYq*L*Ini2OttGJdwR&SjopNqIx;B{fLFK;4>v;xFo}>EFby=|ohRTVT3^{aV6p;~#7AMe4><}r&N5kSM'
'7BIV<&Xa@yt~bZ;oa_L;1bQuEz)EE45*lr69g-(Wy-P_cOj`mD&Z-r^I9NJy%xWT5?1J@3JOJT$DCk;Pdbk}NFhIYS?i=jQ_RW<wlr#0tw}v=W'
';g1Yh_RdavWDmUuGi#n^gQ0$bu@yq^!BjVC!VTNdJ7f|Tz{%N@S;uVEyIn?729bHh;ca6-jOR-*IVGVKec3W=S_!jWKb=g@SZz09etfNyNAV4K'
'XukMnJZnu3S{669YBiz6$UHKd)>W|J@H}mk8Wrq-Dt+FkJ`cYMo_1FY7nBeKOCio|o%v))XS$Tq%Vei?8DdEP_uE>qU+1v0MZJ=Hj&|Io31zjx'
'<f^am{yGd!p7}Rn@Z_0)l`g7v7(9F8uVuPgS7C7W%|q2?jXH4-9`rJWq26R!jUi|y+uj1S0s&gG)(gP^^}~O-Ui2xrgJ`0PD9#9{UJ_I0wrrH9'
'#mLL+0k(xV5h=&gafjY+vm1eq(<Oxl%=)kZFgx@T1A);t212w8lEj=(zFI>lq~izxe`D)eRBYsGvc9^FgT}9ojh=Ox<4*jqxLsKdy9ya+sam`^'
'xw9CQ($=9kdK$R0NP<YnbhTl1M7==Kj8226Edruxh*vA{TuYd=$9A!kzA<dxP*$_J-h5D=!4;T|5?CV!o9<uc8Te<RtrGx1z>odl><mO$+f|ik'
')2rdNCEJ>P<s?~Jdm}2|r1%ycLn{Ww{QPKarPLe8pVNu$q>NTuD&SaxN&!(TLyEJSB+buY{e(&#kWbH+Py-iNae<ptn76@f1d0-Q-+B*#gYmSw'
'X|gnFSpp)U9WaD1TZz6e8eb`O$5S0cJlW&ctF_@a7HcqeniH1Zd!0VHy${HpmOlRZT*qS2)xzekCr#51)&}eu^0iu$MmgLw0q-0@fgsb8C^#k~'
'G_NbnEEJ#vFJnxSqD;fmmIn2QAUhPc5sE#5O7C*$*TWNev!y+&B*n1Pcw)MrN#7{3LH@ce1O2nbG$sdw87pdj$KAm*Wf$0i<E0?Mxi`^Y?tK0d'
'<+bzN_xQn`o!sY^^CDfWWVZu9Z+hNQrgtc1<iq-mEUVFj?omU6VOsv0at6gpFDg1rgm(x=P{)L90=mN-nQvnY9nWO0JS25%7MkrMt#b?yj|LV6'
'y~<WbY&<+EBdN`-a1S0Byi6dLtt?({25lk&v~DK62*drsg|o{*dd;BmTi(PsN+BSq=X%Fj<j{`Nk?e7dn=G3O6r7>l?F`2)2j}hd*-6W+wTPnP'
'T(OlBqM34y3tDjA1V=P-#7Sg83mHI<Fr7R*nT_`#a7+P2aO&#-<H!U7<Yu;A%K_vTX^D^G0(1X4_|Iu_0d7r|6MjP&VI~ocm^H>HB-RKm?apoM'
'&79M&B}rkh1YU@<Y*cvvSdd)o@!vV4X507&tf`vFz;K4rW031MmOcK=0DVc>ENbuvg`QzN$4sKin6V$I;k)!cJ#koobAJifHVkUh2TxFS_?$`p'
'W~BY&r)dx0nJQqd_(|)j-Yq(A!+VI#5<{DFEk*HDx&_@}?V<vx8=ba4(uv;Zsz!-Y^Sk&4E$dik`d_VOav@A!Q|UjgUwuM4$!m1q((Mumr<0@v'
'4AyRokq=eu1V*TgdQM9MfS#cpCiClPnaXUz=CM)xljeJ=N@FB$S<=ov{@;E`TjubXx6YdvF4Z7j!hEPIsDb9{H*I-1gP@nKtP(V{ehtGm?!&}<'
'b<lSFFjd^(IbNUL+(dR0^R;<>=L(S`<3sFZSDJ8H^jmTm+(ntHpELiYFX|bxn;IosXc&Ez74^2I3%0NrzVSnFHq+4{fh;$8?2q~=-gOtp10lHe'
'8MPS80BIsS8n2f%O=HW)=wx&(sUULGtWVID)ib}B%uDYUd)-YMV%_R*cfQ`lg@=6unV>--qdCTKncSikXH7-3>3zdHaCRk#VmZ}1qO5T@0pXz3'
'!`f{o>tQe$43N-W6^rRv=~6EbJgx~2$N<3H@)vJn9_YB!o2YQICD3J(Lv%#@F(<w_JqpgAKE@?x0OO<J<Xig9y@4%6vmrgv->gy?uF>qo(#rxC'
'2Jiv*T`qm`*jL-msur$^qu<hmF(|~kGw~KE^773QW|?6mbiDrf<=2VU;8Zy(b}3elvGT5Ty-i6qstEM8sz*ijOj~os5v8nFutJMKyeC=$t#?IU'
'N^GR;hgFC}R$-yD*W|E*SBO~d;&h?rEMC5QgWVx0tT&wDHOVqlb&Dm5ftjp|p)`WwKpm7-mxRSE+FWw)^^)t7?(1zTpt>NGf;5n*_J`O?V}^>0'
'?0HW;)TCF^l%P13TBfw3U4%pB3h%+}!s@Alg+DkFP-~$&mST8bh+{9!#G$b9wKap0qn#{3Im4=X6Jkgo69<@+0?YIZF``;yCJ-}(<{@pVw4EVV'
'?^I*B(*qt_NaU%7YA@o;IL#=uKr?Y^Pv3LXG~5B>g=e;#!C*o$<!@7}02-x_M^>Ms)A8|lWA?#QGNti$B}dQ3dWQfjjp{@tmOuypC@~hCQ-L*1'
'Le+|>u_(d0RY|Tu{D87O5w|4c<Q<;O?gHUiu{{f$_|NmVzlqxe(me;0li<WZVEuA1na=zJUFm%=IjF4qIedupDeL*4AXXspF?-K`6PB6%&9Yvt'
'ScjJJQc|TjSW;-X!X5`LyeKQY&~v?7$D}RPfPwf>ptlLpu1z4yj-X$-%=0Q<fg4Z?SmUB*&7Gq=S_QP{sK|zSRcpb38pPa`+ZQF4yU%k>BgVMp'
'UskLvKbxQj7)7T`2x~L}>D7TtXeK4m(4J$PQ0zbxl`|kE)gs^cSSlcom9fH|r>rh%Zv`1eSDq-AzKmfu)s3mANw=3&Z%7ps`CR4RVd%hOd##!r'
'es_z}lz^&YG`*G)vue`~eB3Jt=)-o4ku_^=uGILrY_^iqh4h@sZ(gV+M-jaBs+*lery3}Qo;Ln7v4*&7ZF$mw$WTgnC(qh0m1ZdAB=nE4q=HlM'
'i4e-vL+SBmw6g*$Ier+xbOX$cg}DH=`98Vz=pSzpSMh|NAOE?NtLJ=5!uJmOOECZT3FWlq2v0nW6ed{5j#UyUsL}kU0)mgn0jt2#`0mb8%H<CC'
'V4CU@N=sHFZH;_7iV@y=tj{NXRtbV7S(bn#I6kl%MV{f*YTzymJ9y^aM18rN-S0DZK0tBMN(DK3-sA)ZjIX0AzmOQte0Oqk=Ogv=x(Y|9PyN$p'
'&W^C2V?eec0uI1jeN~sEaSyAS);RCcc<Lo<^u0@OMw^>zrtr(mOjiT%E!ZT(rV9hP1D#ud5uH~-w2KSqO`|hXZ9mkqpIF-G8@Dpuj!u1cE_q9q'
'Mqs~~+km$|?gT9!YcFqNHCK=PDOS08T6K97ti!RInHN;|3tICjYE5=f>4c9!H<exSh#bfft7lW`wSNPMy7<<9PQAlDCd-9=Pq0Od2bC7I0fmA+'
'c(e&4$b?QL33ckjoMw{6aLul}mTEppvk!*FAvvfwH7VX9WA~#Uph+Stkg9CfJa4vM2a?Y{anjz}&G;%#D@DFhi6%7kEU-6Q^f}UIr$U$UhN`DJ'
'uN72zRIO35jM^v<u4QWxtx{Iq1`q$J0Rw(+esTl7eeyYVgrFUEH99g>(%n<akH#|QszLVDPR*18?i^9N>hRRb^8+-j%xoXqx@T*X5jKdbu|xH*'
'rI=R8(qZ?I*9gB^G}oJAoY-JYLp!J8L=Gqb2sy|vdgB9|gGbb`+0kX)WCQGDE|atOEZq2X)Bsv0%P`DyeUkSeh4N7*$5lRRNo*~|q&GZD?o`i2'
'rk;|G2-C1gikZ#n>Y-KJAd~eL!`3|%a8LI6bkcFu49Ew_lmozs2B!L^wa|u8w(2tc6>9z>ikqGIW~kO1r*=}M48o7y@P8Qerj<GgsShX3DB)|!'
'=uH}uIu!|+yU~bwI>BM3K$WV{-kKh7>`J|PPgHJIJTRVQv0cT}F^=_C>9K}3f9CVHsoQUnMsY$hd!&WISyM_jv;Uo}>+OYHU{VxQ)Rl6`A0!`+'
'pW^pgL9Lis&V3QpJ8(o65tG0*jG9Og(8011ANS`}{`>9wzePX3d5=kXoKOVh0YilweCUqAuc-(Bi_(SR5(FA-F2MYcP}a7fbI4VqO3o<1;4=~2'
'wh`-fN^xP!G$eRlEbKoGI#7QwkclxVO9VM>9ju#mk{FH-ZW7^{PLVV@3vhtcc8HZI8APsA8)9uCal@solC@gId(i3vLlnTafr-VlyuyYRE-~L@'
'Vb-oC=t84y;7>0^f$qKgBoEoy7(0iOXw5D-n{$EPNA8JVU!S@4%_&*GIo0Z02T;e|(?bJGU3SZ3aH|}A*$tX^!wg&+V;vZ7*Z?&F<~$0l<=sR1'
'#(CsC0p`2GgY-Sjrh0co$d4<a7`NCR55zv0$t`rKHsBo|nk9g-763XY>tq>?oar6%dk|Feonr;uB2Ai#8rl%F3+QkdVl3Ku*vpfu#E*z)<#N<^'
'Vw>Ob>B9iKO>NN4#-1(1ojsNg@YcpkALwQW^1%gGePDWbDu)!#CyIQRD$d62AVjAfLBiw!td(<^+upowQL%YPy_4EG(eOxoz*B1yi!e%AP&4AV'
'JVTw02PDvkz*JVg<ckvcK!Q%9ZkoLihWB51(+6&@z$DUtv5;)QQ{dmS)BwP_i0cebKhwUepu<vFUz?UTI2f{!pwuLZs=3sm!n7=+gS#krU{6&I'
'WjCl%*6BRu9aWr@U7u@jw-3_Xv@t7pXJk>?4pZTK!5Y`UM4jWxY5+KF%F!70*(E853$2*a#$ChX2^#D})sLw%&^_6fnbIjhS+8QND9sEt-fMr2'
'xOC(A2gud~Y_jRoGRG~Dp;fWb7saV##6Fdf%d!lElPA9I3WccL6#$iaH^GbJlN#?9eKUR3knXVAmtttVScalU!Xhpz<+qN-Ip>4@&prNYR*w#S'
'hCJjwgnzx+RX3UpS0LN@)jr~CznrwNP2*EF+eqUR4&ld}=FY)+YRc)kbzJgkNoN!M5uLNu*)Zie8o>uGL`RXa28j<R>a)G#kGG_ia2pDuh%F}A'
'sLsU_e~r)3iuQyeDrDg$rHX3$W=@I6+f@zZviYWun96O8w{doZi7bnpXJ<a*W#vRvHJlEu)Mnlg(ydrYA#9-vnE9t?JO!&_vbE39-aph^2I5e0'
'7r(EH8>^+YNwWt-rx3u(f#^yAO-xl0pvnQPAV^*(a#xAh^bd{|<FU8o3vD+KO~=V)i}q$PMR$P2cse=bw6em(0y4EA*v_Sr7lop|(?v-F2ZBjr'
'&w96@bbQzpt;<ck#nbsp(A8#Gy*0ba&sE}?9@EWTM$$>)Z)6qY)=mJX!$L*p)M@&ZQUBPyC#y<3>-k8$Q-`EQxx)<;KrPPlRSih8XJ`itA3mIm'
'9JgnY13)iqk5mO9PgR4YbCS*FW}p|+Qgeq1a~o@^JsL2YSiVm_93HJx<X$e3RF~EsRR0wQL-urU-aB2koQA38p|rRC`gHmma&erHq-9c@f@%%e'
'wyMq)J19r<da;tttrnqcw)VGf<QlYx_55-0<Z18dZZnJCtrtfR$GC5p<}xFWn}oNlv&?e}H=1fMD>~!~v7W>J6pyM68>-E$B*dh9eW%99bSlQ|'
'AXY`XF4J+5B23RV_79w!%$l=a-SHm=uif2Ht)bYx*CFt6sG%_&dJHdgpSSr)C>I&D0}u<$I5`GtWsY%Ow6?-~a@4NUzqll%xx0{+O7JlHA_~Kz'
')E`x}dv$d9U(J0bO)o6;r$`x4*Q9K4)@pN4Ud=@4lr!mFZnvrM^nG2Vr$`Xod%R|RK5%4RL`H2@KEV!m)Cqe_^`P@QFmh;LZMF~9ZU#bl=CT(>'
'VY)03ouFe+yOFee?)kT#3ijDBQ3*Ok(46h}*gNqi%1i07*51VWK_$#4@bD%`FW+P2yom{!nl${Kv4OwFyJH|x4c0o#kpcP6H@wV@Z*Br&fbWjc'
'5-}=Wm1)^U_T?4B@{jI)98|IntP*hpS3{bs7V%<Ad_c}99R{h^K_b&^G%_CL_?SW15iGWT(P6pxI%*AjPmH_t&e?0>yxJ`Rz7raVnp}wof)eyZ'
'r05%?ska10pxhwZGu5nTcH`~f#M~#+Y_ak^B!q;M7L9#8J5Iznq%MznWFO5V-Uq<1M)dCRbo|&M7%^o%$D}BX-gnDS{C(J5^z;4PDiG4=M?y;!'
'EO&&1j_}rVqD^lC%xwjk-ehi9+3eYTzm$f!PRR1?FgQJXOw&8IiR8DIBGeRUM&q3|Fm8K;8fQ0Wq-H(l?ROP%69s*zJj+q0SeOgSlt!}NH*2re'
'Edq7RDw{Ur7pL|WO;}TNt;)LeVJL4hEgiwO4oteore+X6RB46vN4tQ&+)>*dC6COQ)^f8}!06@P6{p~-$!gKhL+>Bkb1eu-j=40D)8=MheV(P1'
'R)LC<F=J&mGDebOJYZdHs)&JgHD^?g_?e5`%+*<E*2W*T%&ibfT}##wczQA$j@0><yvT$|Hs3$djI0`c>bi(SD<%1l?T>shU)<eM;S2^J>0eXP'
'y1uf_Tr?*B1|f)W_zZ%*WQ_`J;Z|%?%YkYg>*6%iyE*TqkIyd}45;*{RS=qE**x$?f{GnaZu)$^)G?-M^SU}tux2y^pXC^n3mNVa(_Jxs`yjP0'
'_u~RyDmoMAKmFYuP?e3UOQ{4abkgEfN2)KaLM{!&E6X}hTAYkYLU&2d?P+53*1Z9$$^>}-)aHFRJHlLo``I5S2@QAo!d1i}=_*@1+z_$fn%I{X'
'l3{Sni*ujhz6%0otzuYvif&U36gvGo0;&5xP^dp92V`H~xIg97@x7HW?tl;7qzPYyXl~QQn$HY5jt)bbT+r#a<{;!;mhNRqRbElqiN><IYcm4='
'WvMI=Pc3&uKW$tRf?@k2eDEctqJrb4M?a%m({pliwz%`2b2C+R&{t<Lqz;=}^oZBX(T8V;5X?U|gB=rgzBMnRR%jZGJ4|5Tfoc?~-`4+jzkAb7'
'4VZoiebdxevQ(g+LD_jeKfv30a2vs#baSt6vFpQht<VN+<&x5y_X&Ng9HljTTCk7sYw$iBX(EO=qf<$hJi=0L!AEb2rzGtH@IeE<khXPESkFAo'
'brj9DAnf&Z1GsjH#JCm54-{|Ex@erW`q-0a)1M{IS1%LhY-!YhuaUy((r*+Wb<xPcknIVBhZi@u_6#;d?WzV-=^6*2SsU3xnYTSKdrm`A19f$q'
';6i)~v+d`hP!~g~K%0LI%r;VGO~hlzUh`a4i|p%mj}F1lCNHG9<BNWL0iiHh_%1%JvF{TT{M4mfMCrof7hZyN@er0t&;UIQfJF!jE!?JxV>C9c'
'ak!hF27~+v;K*SXrtK9{>!HEpR<*V`8v|UoqPr5|#njW_z1f}tFb%f>R}x&Y9XOI0fn(bCYK`7ulZWpuAHw{GuE(|1l0L2kV!8jOteH3z)68m;'
'H48a&jgNK28R3>K-^Y5Ze5iO{f*%!EN7eZI;G3t-MOzBHu@kt_d9I!k)0Uc7V$!xzl(bUf*_dFHx5BOvM7Hm;@;&_r9*n?w`euCo*ACM5mYm~9'
'CBwPr6i*n>`YQYJ0BKTbf2;2n+qO_@i0X*4T|L{1-SFc&)q~ixM(I1v!EV4F!n3jUO|7LWZgmH}-vPDg*5l$anR=QNzqnqh1&O@K{_7;{kE#xB'
'j&PCZ<xPz1<?EtlB08UyQ7-G#R`hOvtn8`UO4TDVFREVyy*u9Zd;OPt7#Z~keej}=Do(fN@-qrGX;ED7`pms%#QyYL(@TW4nK$HJ+~HQ5AuW6n'
'&10^LXIGq2Zh8F!h&^akZ26?itdOt0$+UM(K7Bn{iSw|Br+EJ#dFvm?khB|3HH@E@jru9QRtYD?0P4O2b@rapRO*lLs04!@sPl}+2<RPt&+GD@'
'0KJcdpO&1Wc|RmGr9ThxMDWvp>#XKTCA8|q8?x{;?>Sk3wum+&75lV2&%ThS1*3&o8RIhSJMGe{JK+*}9-Yo2x`#dNH&$hnGI0tA|JnE<s$b>C'
'9$IkHD1J35-(kxyq2@oP=EUmW*6p2A_lKGf{5(L{xcGGqK|77x{(PP>z|XKjaE=UoLA!vT0o#C|0X=6}tmi@qQv%eC|4mZ?w0E|Dw9Z9X38s4|'
'q%{JgMyi0ny3JO3PEwm?qaH;uway>SESIjoZF3Asw{6}2G3)`3J1VI)okt*IsxqZ>`xgDJ77hkL!Y}#nwcs7%S$vI=16736xxe1I(?3*=4@A%>'
'B-Gt{B^JdDUfkp{+Ef;IKShlDdf!|>LsPV;#2uZ&GV6KkG&LIe;JaSUMMo?F=#cpzc8C<=TzOrnc$U(pxMOe<e0PF>JVlo4JX>_E3t~t)JRXKI'
'#^7z#Kaq6VJ9El)oOG#e$TiX`meeZ-cr@}0QoUx}zMH;O_V!kvU#EZk;mC{Q32@u8?q;AJgzh=a))T!MZ%~Ijo;d=ES3A+4f9kl9ZN+bCrq6Lm'
')t9?=nEFu$q2+~s-|K`n`>dApciX{C{gJI`U`_7)o<FtN)v2H3&XV<#yf?>*c-|#w-5V7&WsZ7N3Y|x2`CK2v>~VY*6=+YP$1wN%M*okT+mG_3'
'+H(4Z0_b6X<I8v$mS@qL#689Mc<U1}ObMb;F7fX`G=GrnxVbEx%R`f@xdXRH&|3SM>J{Isscn}I^mPR|-JluvT`}M*E9U-19aflH8O(g=nt_GH'
'GBjN{t+t{Rxad}Wp{=N3vwLFYm(?GiL}@!y(SsW~_!emQ)|5J<6gIGg>gYH9LR>hzXa605x-_buXSfNaCfEOW0afrcBvGxi#MaopIR>nT<)gs9'
'zD&Ws6%rSvIoFNjn!|&M#QJPBNcb0&msx&=w}P%uPLEBg^AXSIKh8XDr+Crm`wTU^eF;OUU#nj|mPzG%wK>uql$n}Y#z?=itP|cy;~OTh%~rQ*'
'&(JO3OLfxRj3-vP&`23D^_qU*TXWYTRvV0vO9HHBxR5vH$lUeOb3tZr6`!@mSmgM}fd{5L)%AWezhw`-iTctrV*46~39$iyQ%+FpHQjpCh0-}d'
'+Bpi?Y0(SCz1Uzf&2`<s`@+Ayv)o?7SVusXQ(zsxj}!|l&p`beI5Jb!W$0;x==myR(L69)(5HdUEQEvu<#s~_ywBCrYp>f(8&;WVq#6}+&n(7h'
'CNHp7$h1ar{y$Jl0|XQR000O8003fL*e0S0y=VXcXOIB^6aWAKVQF7uZ*6dIb7fy;Zf9w3WiD`e?Y;YZ+sKh9`g^`F{{w=Y9FjIjk+SSKoN#t!'
'Ta)O<ueCgr$#QfMhy*1>B*6fn9;4;|e(TY1G)T%ble2rj%VaDP=tp&TbyanBJ;E@2n@^H?ca{}p6&#-ilYFtt%dE=sWsoi}vt`=*^{;;YtIt<i'
'87%VYdY%T;be>(LMN*~n`=Cr$34Seu>Z(XnsK1<~#j+e?ttUZ#QKrRBg3S$s*=$wh7ikof$znB6r@^d97HJs;$#NRp=Ebx-PjAwBP!-9toD|s#'
't9^ydOtT`La3e`jE|U2?m?PW>5>$X@0ce0B399=QH2hkR;^REwHqyIvvgS4_V>CG6Y)<f}!A&yHrZAdmu!3qql>=x2^p@Xt=UEBQfBmZ<_~x6%'
'D!mMn;_@cIr(%NA;O*OY0W`dXK@`Ddc9SltdU27#hzgjis}umQ@_bGs=5jEw6pEI?DlHaSSptB+{uOk5TVz#=^+k8HY+hlrMY6gQ^D)1NuEQ|='
'^{-|{z6j!YwyxGi8plBf!!IhRz050`{1Rc5Ps{t|B+J$B%Zd7Wm6TVoxaxPdN~Y5yElc&}A0@&ke$4Yrz>rJzdy!OE`bV}%aW!<^Tz@T6^|ic$'
'RhjFb`Q$pS^w0GLtnS1Fp}g1MtMsnAg(=YwpjR;;D}V+DBPZh{0Y`5DEC5@+I9}dI!E0FM<YF%Cu8TQT?E!9=sVoQ2{*h<P2*1z^oKq^DWHIlX'
'G%hDs=^~L;+K4X!>?=md^i%rJby`-D^B8edJ}b+#vWvY+%JeL~T%=3+#6fTI^iPNy*~EDADJ}E)O*(y>+^2<jb~d{^xzSC1muHhS2SgOrI`tuC'
'g&fYwG^?mwFoBI*B=IaS;>kQ)MRFU(DHMru^VHi}<@0QEFWcf@F)Z$4RY930E@cFWK_F8*d66Y^QLWzGKi9BXE7e}MELX56Gysu$RHVykTHtCS'
'!onsRNzCnCrqx?|)(&;kVF$~8clze!Lwxf6=actmryt&*L9K_dw^$v8Q4scS(~AXtS>!kT_geft;=jpFMt`f4A0NV(&98rT`tJBoC$(1pvAU#E'
'|A3uNUslUY+0BZ+TxK)+d$CwyGsoY)IjsYCn+O<I&jc(9{e=<I-`PBwFmUh}S~>gt;p1OFe0oFG86U#rB41ZnD47Ahp)Z$7wxlofGN-@GDg7x@'
'{Bw~`l65J*rc3^tA^f7ni?q5GFDChVPIYlH@$Xe~Lp8HyMNKbqZgi2C1Vp}^-{arvDydY1K&i8WpQa1(m#e{-Qhdou{+r7OMZUhg!mn$@2K)k8'
'f3F(7r5^AvgL*65yoEh=CBDr0NmlU#Ihy-?E#aZT<%Mb%UeP%7TW;c>8<@{&)QlWi)e^uaOVNz{OT|U&v@XB?)drFC*T0&~ld=?)AU!FHyl8(~'
'FOmGwkIs+*SS4l26{hJdh?n_oyAyQ34$7(!C5jZ7#*zqU51!8OOFR7C-@3nBbbmMf{JTF7fA?<qyR-iU5U9-Mf=YBcMzf3LE-nGgNg7|=gD}8-'
'z@plHK(Y3-!{A9U=pP=5Ke$b+c3$FEOqVz9@c1--{o&om4`(OwyW{WUvk%{XdVLbV`rGG|GXMb~Y!6<Dx=)@w>vSwYiuJM$yLbU8Iwa;ih6?8l'
'In3V>JBtFK^yz$vZSh<v9@E45oC6@!kNiNuUlsS_8-9oqEr8G@F_J4Bb}7ja01)#l0S1C5D?;h@yn@YED<y$ekP09>Tsgg)q^l}8r9NnVhn`lA'
'aTxP}b2fz*N5<=qryoz~d0G_p=K_S00tI;5<a+cC?Eh2%l+8r9Jzm-g$YH^6MnT_zYnoO`HpkI&N$##UO|fu0TvxO1OIXnqhf|KiA_b~F0pdIE'
'9`*bE;khxwBm*{V;rc9mpb6RpGw39R)qddq#@)dSti1_47Va?+2C7I|FJ7%LfWySV)!Xo&MvM3PGF7_>*!>iSB3XYM@eqD=aLZ&X+6`h}xNMIE'
'fiwUC3~&yJ^Ho(5CC9(O34*+!ccL_qscH#4b+~oV;Rhgx0mB>o`Sa(G?Xyns?WeaO_!b~5P7q1bsT?@A2_QQ?^gx&ecGu|@40+fKHSAP{z$Nke'
'N>E|-qetJS`2sn@W(TluWi1UNj@Sj$-djT6gbf8O2_)df_~?AxKM#PW2KW=9V-XPv2A0m2v%HPnL~^JsY!<=M(X)sS1S3)7?8EE7#AlyBogBaG'
'Y^<Fnn@lo7N05>6&GrG#!q><EXi!3Ifxn9!#deh9Xf^>a@eg1llMQH<8^r;M!Qz<IxQ#xu6-<H>!F$;%7GK-qZ^tM*gOa_h1c54p^Hz%E=LhpV'
'UtJ`VYvW0_ypH+&q@F=@a8LsrS5FqOU9w41R{GJJ1gdP7!G6<SH?99i4aqWJc7gv^C}E4qX~YS&LCa*8#&GOO5pC7hjDW7p5`Zl(*Rxr6M-O41'
'Kr{lJba)#Vfww69x7FsQjf=+v*uGRkZVI&#L`AWCYog$P1vNSDzXDe#Gp0zvVZ&|_DEUZ_|0+3r{-TX=4ru_+;HKGSN;_Pf+Ipgb7zIixbO!3>'
't9028iwl??z-wqAO%^6Qg9Vygt(Vt0RM=xhd!8&VrpZtg1rnc3+Xl<-M8QQEb{rI7aC++%64f>}Xj5QO;3~c2Z*8&IbRG?8$kY6GInR@6OwX(}'
'FRN@R^awZ?QSc-}d4lB);_I*kNU~u<@Q+KvZ89UzyFiTZnpo^!M1bh|9Ul7&IJCjBd`k0a(E&NCvPJ|Z>jDlZ6*CroG|v_pK2T%B{o@g^xHQil'
'Ej0nsJH`jE=NYuxmVA8hvft}R02eAUV5hV8pcD)phdH4N4o<{~4@I)POxrKq#rASfF`ABD)BusY=`mWh!#|yT2JRwiMZp!ISqhTCL-;MwqVDk}'
'JPn6Ic=qT1pnH7U74v>D==H-5fel1!l`qTG0Z!~;`4;+1s&yHYjDpPVS$_}(&-#b>&olhz_xR7tji(t}5ulW$4;$$w1+2(}+Mz5EcKi(hEJ#RX'
'v4(D%5bTqQN^<dV&qJfZTL~Kj)C7TrXQ;$zt67&&HPwWJdA3}q`_J&#qX6W$hCzDSs53$&7O;&_z6l<DU2J^A1hI0S6eOhiSBX>%%c=|f<Py$r'
'o+e;Ksufhw;lV^c=mcN$*4%$!_PCI~q*GQi0w}ZWa*h4qZsvx45LJ1Vpak9bjaN}bR8!%;T3iEiwE0Il`dqA2kgo6G5ESRv^wZM=?Mt&QW)`=C'
'SpfN1t((ToJKJI(YU&eYB!X6%tYZuE^I)Hj0$3r7fA~JIR$PAjvdGUbvpn*{v(zJ%Z6oB2*n%T^!1<i#K=haQYLAxzX&e;$K$Rr{Ye`9pWqt|A'
'90kt@&Ut(<zFDw>vixbRQ?~Lma?fpiYfqjfQJ4q`14snKUQtOu?Wd1oVNPA<io9(d4(j9EP8rGT+lRpEf#G+I_x3U2r?SN_W{11k;Lpwe9?$mb'
'>`sl5fFq4JPh8um`V*MXE~cY*?CulSms2zkqxp#p=1UlhpOp|N5fiwJmk1oAMjBW`vqZBO12Py-)6)DhOC&#AnawZ6WYoPVs}00aq%BS)d_t9K'
'GvqZ=?VIOt)J>}jTO9?4lK0)oo73a??T6RLZ>5I!&FLr9HHAv0rVitzU@@B7NAJY?S->NhE4q}0Hqs^PAEsXgIQI<@bTs5rZD4lY>7{oo*rw}}'
'b!e>iL2Y0*smv|FQzM%&CTi~%jDe%>1$SOYY7MzP*4xZsd)=gQl@p+zBB(-XcXmvg0b#IMmsM~9NQ#%v$Q#NGSj6@TL!Alth}5x-ix`*Lk8m9D'
'9bWfZK{17LlWe;euoVqlv8G(8(s_DWBnu@nwQ+cHHf@raTv4d5G^q0HbZKMu=aaW5fBJO%E`IglbNu<kUryd{!D_0%V>0RuN_^HNLr%b6>VTtu'
'ukSzg8=_Bg3lh9Mup<kbvF3VsaFDFBo}9~GUR)kr<kbU>WOD!|zu{XlGz(dvBqSyo#31o);}N#Da@SRIkIQ8;S2RN~i%`>qydznh8855-0iK6V'
'ho7jT4i{!^yiqm34%Kc_mG<DmpT*n+(@ZLd2`$BTGP~wkcGX^6rW#u?=lekcwh1i^uIbj22?^MebPLgTShr0eieqd5gPxQ0NTqrv>37*rBfs!e'
'787AWj?0YH_aG2Ni4?m{0N{bgxIOUBXT6F$RaiHj8?7r^AxT87k2*DJs_XBp+;!LUJ{LSp(7|%OTETXlI>gRsTQh93v*H^@!F!AaodTm76U94T'
'-k%F7gl!hyB9c8$kFLF8xA61Ord#6ajuFT<jpLmLUffyIjje8GtZc->F4;2Bbqs##mRE2H0%}xbX(?{m7fCgmt_vD`gxqE*4pG^1n%+ftvYEtq'
'0TkV4Q#b)%rP<|G6%{cVp2cXlCJNydL&Roj;_+3oT*4|**E!i<Ak7gs&^)J|bK{UTfP<I*74p*Y+==Jj`!3ie2}}dF-J?@(Ti=!fv371tJLZ`~'
'3><3DpeCFepeaG(XlhKNCkr!0Lk3_S-P_+K^EEG96nusW6JH6e|L5$(`!@szTNdnhxR;Bs^m*BmsDeQ11zz3(z_mj?uOMFk>G6nX4E$JJ4BvoI'
'l0&g&dRMjEbT?>S5{h>4VhkS1W0=GUXlpbILTV)J&`b%Eyq;!1xmjw&Y8H=qa>4!x=Pb!s2!I_^dBy}loO#zJ-J<dE861ZsJQ#SW7JjJuH_;R~'
'4GU%AkdK~QkO&J41TxhNA#E;EpTOec{Mb{Tw_&}4c<!k|g-BB|foet-y?aBe!faLA1<gHg7lmn!dA)|gMV`;w^usDbPgM8xtFilyrow<hMKY<l'
'b&C!o6-f%<vb1v}wKV#{E?IOoBdgFlxyHveYK;$f?pJEiL#yi~8cQNwyl29UL~<5B@Q`}XW*d53@l!Q4^&+#0D3N-ycj~fr+JL_k)kj;7KK-5y'
'$(C(3S;Xbw1&Ns0h{;+_a*U=%P?xQX)XsYnyzB%|kyh8&S1uQB7EQ6`T1uN_V5@ZIPN(jIvT(6uUFXU2a-CeJd~3}YmNkis?3QdUE!lJ!qGpj4'
'21QCXpm}lMLJev@C%ffYQdViv1#~D^X@OSN*YF1j*F}zYZ&Ox{lsz&V#um};x*U4@1niQ;mTJKv7>*p%l5Z#5&tP^J*|I%&+23Zw_4(8B`?J@d'
'PCtG={qR11{pWAr|0O;<c?~?}j8^!tAF>gbNqwaRXqkeOeIu7+B}j8Y926ZALai6hXEQwbE9G#X)Zy&`1CO@Ly-AwQ6}A9Skv(wp+{!NuCX7>I'
'mDbB2bVcX(iNk3E*+OsN9560`zF+G{K9XYN9-udi1ER;s=_cYiHXsoSrS%f+LfT}5zP$8k^qOQgA`0bw*<%ZP<W9I@$ug6T-1O!VyH=1``p(^V'
'^xi=7AJU%hvgv)gSXKAoxufyMwflqI;m%Y(fh|rO-ByIt4pLj?uVRl_s1y42=ICW0Ey!#PAXKv@I5DAUHW8y$b2~-*IGEItM~p+v#5NEwjTRj='
'An{{FqI(<PyQ+0GlnQD6!*eANY5|4cL?rm(yf_HMsN!ofT~56Xrep&Ah~{`lCaV?Hq;_nPg<jV=Cp(BZl(t4xHTk3+oh{Z2dA)V?(o309(~U}y'
'?ZlQPhG|w9bcP`codSDn#wS39g>^^!h{>mbxQPlur$nXDg2R4>wK|$~qIw4raSS6P>cU-(f&ra2gSJ79pvdzM?2-&xYIDJEFzB$Pt0^SC>6{yb'
'-OVX}nqb1%z+Gv|&Br+&VFiaYgfLyM7vzLQdM%4I3q6otz5@_M_}2q9HS8ZvH)M2g+<;ieGW>YR9DZu@&lO0S@dX|i`Jg(_FUgO|Ilf9M$=)k!'
'Ok3q6G50rGJi4JHX*o&e>FDTXf7m^w$N%IqN1_SRu(*U1KY!?Z9-;{@+5s*rQYrHUdc-36so(FBxf$U-nt<bV7Co_5!k*g_9-kx+4<XJ9NZJL;'
'ENwMKQSc0BgA`~(Jw8*PD9*Dd^apXhc2>m?yMcf>Q7nNfw+%Z^F<3ElAQZpr9nJvhe^-VE{^8!9*aS?XthzFp%+q8UlbWu5gQsi5-rTjlFVbGQ'
'zGxTW`2Qr`AN$=uw9cQB);J2Nsx(iOO60H=dPLJz#oT6#L|3Daz##j52PkX=4Jy$e1@BC8qBZ5m)>)&iRNB@7nfW1J2kfe5PayU>9t!ZzjfuQY'
'@2Sl;mJq6XXB00rB}a3AF%Fu#(r^PXsT5inB)5Fe*qZ@W08s-M;JEvrumtDgFD^y*{NXUdor(Rcl`t?$UIJK44vpKw9+$!`!QX^wsdmAsy%O--'
'sLK!KPNDL4KXZw1Wd}{M<V-b6m16q2iZSgp=W(`-nH6kBD*MorqVENL-^E}>-(!-0OhHH?5z8I;vo3dh&Hd8Ex#etu=Z^@JiBq?*;Q4J_PV(ZX'
'42`}?Q{!}=PAW11@XnoRirTQbpsZQvg8;p8&qU1KLNeT9dd$Wa>sYfXJ2-b6V|2r!{4HC`ml9#mrFwo@qOToajdPifDVe~aK!TtViovbc%SpAC'
'hof@6Xogn0B6NDtOo_Kw?RNNY`0_XS&oFElkCLgDGho3>;Hr$pI5dqA;}8{*$wh|Vg5kNpDbnY`xd%o7?HFL1b%FV`$$yA3Kng~>e=*P+KsZ%o'
'Q&4|Ff|?6I$!q=I@1NUTg18}yjL%6J<TXM0+nnnsQ1R$FJk-rb!I8l(W<crCAAp=Bx7`EL*=^`t7ew0>2DB+0=TzxI!_J?hMh8y`iRf4KBX84b'
'qw^VC-R9skebz=)Br|8KKGurQb1CmimG^QgTB?<d)cp?>QrZqcms&}wuDD4uavK&RJ)GZtdjY-zGCA<`Lp=c1mZlE6`9F?iBTr#4ho)-*dj=EI'
'8=k^#Q_dU-(j%S$fTI*0!+;CGsdXeT|CuU=60e;*s`83VB7V=O3polB7%QPmFYv1GDhJBTC_EVS41=Nx5Nf&=l-N_PSMzi%NUQnBI#5v6by4Pc'
'45NYQ=WM$o-u_X{Jj~B?hu^8LW4-z`1ET6vdV(X2#5{L84J|0%+vd8F-FW0<F0vn|Y^T<VfQFIj5V4tN9ryI}=SJaWnofz;cokRzQx2%I)d}#U'
'YpMs`0M%>jQAGjBStEA*07`R<!uRd?kha84T9m1)4N_d|v1@4xiwuve>zJX%Bgu~A!8xLSjt;))1tzNE*RSQ{0Xv7$4~si^&KdHZM-gH4j-*<('
'f*tX6d)jlfbIlCh>>TDp&--}Dji1+Nwug(<_AN{Q++5T3a>@2peRH>&bitW@coj>E0AC4f30uwSa}6C*^!ej{1b%g8*R=`Mg}pjW&&=T`K^`_Y'
'cBlqU0t_2fwcT-Nm&YrYwwV?hqu2CMcP{2)ME`0UDGU%}!FEG;@5*I>&1(##t6}$`2}4{%lP@CZiliyc0@E=)0>$_ZI&W&zO=UKUF?4v-J-}GZ'
'C)|y7H7WoX6Ej=S=dn0TdXK14K!;B*El+tUKC2U-WHJ{;27=2|Jd&b%TQ(d8FO=OAFriqQYYUQOhSD&J5b7yN<miz}MT$Tt#eIB}m9T6jx-p0O'
'Y!=eNLV|!cY~0HngK3oXQUYqMAe1id&C|0iGaf!OcdZSvL|PjY%mH{5QPkCl{`Dwx7*7x@4tviH@yK24ZZN<zlBWfURZb0V>j)cl+d@9|%-<OK'
'cH>Ns!W9UO7^Vi!N8+oiSfc$TWfNE<po1}x_ZUrx@ou8Zp`x$SMY>F98T9ckx#n9kQHA3Yt;l9>0`hc9Msfguz;5SQ1dFeVY=NPAmPt{N60GOz'
'MvK;?e(%MhZ|-h{L|>${ax@MV<a#5BUFg%!p#f$LBqE+*WEQAiu7FymMLa_Xd5oixkanxA)iyB9RtvaG^>r{fBpgO&Cvg<d)XyApv=5@<X9u$4'
'hY_~EGoB4!ocnMFKLU;MHrI!}blQ2@RCy}~|Dmb97cadSj2{DwFY>Cw7$*LLcO%?v{&)BA?)kHZmt-_C8Z=gB8wqIP==TjJd>bdd!^2+UQ{u{x'
'+B(P6X^;e@=juugFxSpuW8ekfZE`Op@9Q*O$*m)Bl#oi|X8<%cgY%1b7vaa+NzY$2t)sJ>7TI(K7*y@Ko$yoJQjJ>(Kb_jH;s;lGel7fq`N1W}'
'T#_uu2*KoV9L2P(qZ^786INI-Z}litQ8~PtXH~lh+yC~*(YW`Ue>?B|P@;lZHQG`3NTLN(;-k=RBi$QH&#0eOz0+AuwPhPB8>6O%#>2sLuFvgK'
'T|FOSv{ZxC|IBC?InH$~XG^hIht|mV%~Zyi;07Xd4R|mL5r8I~W0KOmKO3pWpw~!gaN<VT>2AUS%&Q5qWC`2{`-0s&$6F<|Eo~mg{X89o7&>8D'
'H5IDKe#Va3Hb?G`GpY$zw&%W)U)8ZnARV%r0ij5ZGmV0mz5X`r;dm-(K@p?)yaJp$bRE?|YMfrC7UmDm0e@fi&wT}BQ_19&k|KR9_wu1b_e}8f'
';qwG7A=Zt{%qVS+<FUAH&B4>)uy?o>)w%d?WG6Nf&(m2W=myTCrXiW{z;_Wr<4X{pf&K!N=g0}`)gy48tjQbj7nNm%z6QenHaHH1>5XlW!rnM^'
'Uc^u+)%-pX-ULD>OQ!*M&^23AouQr;^+P$-fQ3OTlOkXB0%=Buen_n{074V+F)M+yjoA{$Zkh3sHmI_S^lj-^B6?-E#8`sTM12D7FUp>-O)bV;'
'YqX7%StmVpK?t0R`He+Kj7FUuFBp|k;Qv$n#TeIrcQ#{yy6=8NMC5vp*k<MIE>3Sqi^39^3~$+R;Gf<3WDknPR0`ujN0;CzXH<U@hh&FsMvPm)'
'j1r>*Ce?^q>IqfNO+H^Q(viBonBDdGkyg-f&&aIE`qsW5cdk_o?4(&9*~${wHixV)4OLKIAQY^eH<1hxIabok-kN=~Th*<|jWv>AjCU6ahD8;+'
'p^^Mz6tE|JWdAhE>A4^2uf}^R(Z|BiQ_Z#e!l=Veuz7(;gA}mPUhVK3fzKQcc#Tsi!d;>zK8MhfP2j$!Q%P`(BvfAA)Q&#0@JuR#g%ztdXnSyI'
'!mxKZ+m!VdPUPj8Ky0WfFkGrB(HUWLAfF-=K=nh0FHwD|UwL(48?#Hf8#`v;NpjJABZe*mQsLD)^pRC@j0*IbqU*lv#V3*0vd~@R(8X~lpp$iS'
'n$LQoZ5^k<<XYd+K+ujpyIdl7D;1I+5QdcpXWx7uarCh{nrhw$u%n9{W1tDR@}=1#D$WI)Tf_Db&X&oFZfA7RXmNd80%{}}ta-5((U(!-4?UU9'
'p@Y$7%Au%%L|mq5#ZBJI*pX!QA}FGHz=m>KJLODCY|ki6jl;#98SYX1Y~`-kwyTS`)AeH2#;+rGfpVo>RA<}W3jb7EJkJ499j6Il<hCJ<aZHi$'
'Bj{h-T=U+1zCb@doGZW8`P;Ye;%`5_WeWqR*xR?q?~dcMlTY8BeA0!KU6P87co5o<(sVo;Ob$`L70`Dd-kiKuffl=xq8KnqIQ|_?>9WZpt*-K^'
'0UC!0nL_ggh!(&Ivk8z;(skT(FjjPJF~AMO@EGfWRLE{wR$Zod7z!8O8<7ckEa@Vv%3!?|o=C#nS$Px~v*91|@!Pi_{u;kI`F!%}-Rb+&v(Klm'
'<4-5AKfrSQjXe7>zDd}TwpMiBa|kmTK~}|@w4wLPAQwLyYt3*Vd<H;&rnD&A_Lec|9EYN23p0bOz(X`eHs0EFNm9pV+Xpe<ABYP9QOcHG5pXs5'
')5mWEWH5NYf>Ov|lFQ3EhuoJnPbq!@QEsjeWFcQwT{PC=pvI<a{GY#JZ0ZG~Ekkr(0F@~fwL?ykki}0ixxzW7xDax&nQle4p3jgukZj2)@X3$&'
'FnP?6iBc`mb!;vo*LN|7b3oB03-<L}t><&AwFKHIfEd1e3yX$Q708A6d}vvui@dn+Srci1Q8U_=Vc>C44%ei}?*=BQ`0wi~e}{<uO<ueP-k!|g'
'zKiJTXEb!me#F?}vXy;e9nl8xIfk2^%T?E>xQn%b`@n`0x0(_sYW|Y=VLF{n?MMV3tnh^*pjqP^N!iXGWAQYOkg}_W6<9`0A;4x$X%B}1yDEjj'
'X3KQimYqkiNT{skAW-CWXN4Zd4>@;V-vZK<u7MIoLZ8Cn0!|i-WQEa}66iPFSfizR?oHOy1S4dq*=@T<14c*VkSfJf^7F)gxDm2)7`!l0WL}F%'
'xNkGh4lO7f7xV3h<2UiqtJBYl4BFJPP9I(6SbgVg-6OAn;vX-kuh3KUHIWFDM0tn#&vgO>>qjCaF_RS<ZoR37lQy-ME#sq$tn!7$lAnv^1(l)j'
'{cAuOn%a@II$PJAWfW@}c6CLABO!YfE@wxLmnyP6oEYGgy@u|%YSVZ;@qUa+?X7YUOHBNUSTwe4?PmnfZIdB9g-{NUx&Ub)g8&&c=PNk*;G-qG'
'1;C!2WF?(f$Shvl!sCZUgprr3ZrSrso*3ion8<}}{4!mVuZIc)Malw$C3^%i0tR?2F}+bRu8;n~zi45R&(ZrJEboCXH(v2cix4Q;fWTDhZ~2;|'
'bBlSCp4)8y*b6=hNokqtyJ<5CPq?LR7Pf`aDy}lR>PJE~m(Z`^j1rRc8k+Ixo7}*3G*xO+aOOFcgB-oF7H(BX(T+wG%k(x@7Rabie?B<e=01wt'
'<@o#f{mEZB_zgRazU-6NXzd_!m6S<U6>Y_KFi!;>aPMO{Z=vzYVl^iZaL;BSFFi8PQnh=Yx($mhQLRbmHl4--i;*7kbZw(*EJ4S3UZ?Q*E3fj+'
'AD~Rmf*29y*Fd#GhCvh?f|cpfi*LOu=l$7ZZrT*wR<^8bj*afl{k4?;F83h!_m++H<cS<ja~T`IM)uDw^`?2uu|K?_73Ie>0UlN_Kc0Enb4DLO'
'H>$7yPIDvSCaIK$gz9_+Dn=_`rQ`14eCWXu=@g_kqKJc{8jXVTI$OmQk3N~pQ3)GWhaJ}UwaGss-&X7FSgWQr(pb7<!rggwtqB{(uP4&C*UUNH'
'bAsw}pbVl$`Xt48NlCyi11>X;0y&ZPS=kEDMdwyGVtgtRxYZw5^^+Ij)TB4b=cqkx*J=O0GCd7jm!%_1JV}60wrzcnZ}5%U19w)2Dy7(1<e;H{'
'|8?qy<MNtR3M%kJUWg=mrE_~H?;Vv(26{TegVNGNfS^*`2YAY7#gKsX01ZdPU^t-zUh81yh<QLMXma*2qT2)A8VNyVrx%>zAsCegoX17SpgVs5'
'2Jj}G%dCJxNr^JiEKR3)em3R}rXKCTF@+K;P1B@Oc1dW$WNWuHBL*>-6|vJM6@#|e=P_b{Ir?H8_$Y&o-_a01biGue{9r_%7@${<gPFe#NJW4r'
'u84O=SBT@BBN4SlI8Ey!)wwfB%9Brfc7p;J0iK!!?T*XYPyX|vj>IR_&7|nT$bF<|6R%kZ8dI)~m?HIwPF;ZwKd^B_>xw9p{*R)yM+8^3+A%}f'
'e5GUXSu(TG{|B74uOTM9SF>)YxSOJ*%iDyJw}mF<y1y|*A=(T$G0s3yAXYaP=ugCUZJ6LpCiiO=`;1i`QwNB=5_Ivzg9m)Mu;VsCwCj_b?pD$j'
'qUjI@CjzvPfRhJ*+eG473_Q?!!>Q4z$hNkG`G>E%Gz{6+(1!d0E$zNI!l}7OM}?~GLvl%BM|$Ml;6W@^Po4GHIfEY32EaM3uk6N)rJ{>R?3Q-|'
'D6N64_cbJo`!G_z;OsXfI`a*+;RB6)6{Nsq)c}_i)m4q1GNR^74)#L!4h`EeS>CtVeS&Atv{~G92Aej6Z<bE*3dJ)s>?TjfIZ~4jgd4qR$*9=}'
'yr4mx9n*XvqwtwMM}apcpC|gBF*oYqcQ=v2Cz`ju<f1Zo4U(b~tM<5JJ6lfX>nR%QgM=x>|3E%m7jq%GZaF;EKOlsn?SE2EvP?E9Ou9y>$9dEl'
'%NRk)qxRLO!ANtBStvR2gZQ!mQXm`hW)^XyMaWxkOCllRB~onucjdt|p&k>DA_c9d3y+Y|wk1KlKopD`_EN)HF#KWp`opJBC$BNoR`BWMKfXQv'
'bn>Pb{4HM>bVHNnYn?SmN?dvvy(Vo__(`(QiAazr#YveeS!kY!o34@y_BjA-Dxeyh{wPw9v9KmhBNVeVy>QTig{|6Em~xP1r@4r-y8xcm*v$+?'
'LXw*nX@QBW(I5`^qdoj@(2RpNylsWT#@NUdF%ouD7wn1?P+%I6GIewU`2-NK7raIbAq?LJVr{algsd%8@8%RVb)%+c81&J>j{9NnSM(FcHr9(E'
'pONW}bjwm!CS~(de8%*tBv`)s@aAvwkSqP5FVY$M2r5#DQdC-0Wn{z~mxY`Ac+GedP!4a(sVxK>u}^<key6vUl&s0^OKhL<5Mu(K2TvRKz{eV<'
'w-?DhMl`PqkoOaE4*g*nvXeXtrsLrNb543B9m{#P&K(pj5POM|<_6D9(jfM~WTjt8R?1`R?7Z5qi_u!+?G5FNAF@P2M^mohz47DFGK4)GbRqov'
'-~V3Y7P2K3qvroWsrJ7O5unb${ZKwJ1J$6<Z*p0VpzvvDM>|jzTKZpRb)$;*kj29ky){;6K@^4?OA|QCV|x$1CqHaFkeu1W=DN5o>09rz$k(fO'
'zk`N>9H_@RL)ARL?Ru6vg%^vKc4%C{SSNU=^mK@eS9?dfFl5&h98SB>?{`!aDNBvObgBst3-UPR3bDi&1h`oWF=l|o-HACpDm^NmpFC%$Ew_>g'
'1<>PFqSKM{ub5NP`@}7LEt0aRl5H2$woRC9;)^fXJT#+$Y`wLO37y#SY@>sVFl?f%6oHju(5d-HZU!>Zc=#M>+s0GRG$ycCiH_hp9tzJ*MD)fb'
'5X}i3)L>K$=!8W4{Nf+T0Z?u*<t-^5Q(wENVuR*E>I@vpxSd#rA#kOXLo64IIojov(D=N|cbOarnM96hHk*O?KweXV9U9_;r8koFjzTu-ldvxE'
'+u$=goN_W>q|+3541+w=U7$!^gok$$P0kP^Rtl8?>(xMtK~pTV$-fW&j;sFna1j1oFrzX}r7BD`Cex9U1yL%*%AN!IIAo<rEGp66eOO-g2M0;k'
'6&A(+J_zk0{l_f^6e6*S>=)To3>7TNJfVo!6y8%{0OeR`fra{leRN&oe49-sN`r~wLNBo7MXBJD-n_SHBDs{~#Vc}QwAQoG+dysRf*d5>G6oJg'
'KB<fDG=!0yBd5xFozWT->J~=MhF{OcjuXa`rs|N~NcM%+w>!6VZ5;X<N=2UInvNR4%X5r3q*~F6NqYm-4ZtA*wtAI!an)d*V1FU%f!Iyw1);_#'
'dDLcl+tggPR8H<+$`agBc)>4e@Y|pubEm3lrU{^i>D5!LlxJh}w<A$DFQ@FpgK@q-eC;}oMPnd)em2|CqP-5%|I|Ra+Vn#g^+|Mi2e#8ZdH!r%'
'_AD7;XHB1xh7C&fIe4vA`doV1)?S?N*SYONyt8+wxY51)#_8WL1fiZ>drOBRSF;<~&j!ILZwK+UU-_k*wvh)ytD>66>vTrJXpHMI-itAYZ7u6+'
'-w?fLNGZ#nRZ<g6Hqq!pSY2ftFYcdfvI<Q2m-35*uN4J|2SzfvMw&y^p`r{16i;-XElI_Dk)W>ERq=+(b&8-M@2MbMgTOD{7oll#&CxK)=OPq9'
'tx4r)MNNX_ld^U5l>V~o)q17Bvl|wG!kr6u$Xc`Hkwr}^U&uiDX9c0qQ>nR<b7Qu%*}2sX+WAXuaz*ST&TvUNLTp6A+<%MGdxL8x;9%k@?~O!b'
'+jfQ`dkwb_G|vJhclq8@SrdB{o@^;^qK3eTEv4m(?{a?KcrR@J<;1im(i2|UX89y@Hs7W<`wbaN0B?Lk{eD|qPXqtAgtaz`J*nT(bGi_i2-7ge'
'V~Dsr^r3mH<c)twGX!V_($IQA0cv3SRhVQN!kB-ZM<PbKHh_Q<X!fz;4M!0@G8wRVJUl$&7?d<57C?_)nk-4Uq^T21&8H~znJ|*-m&JUk7Php('
'pYxTt*S}8h`7%3n>6~@`h}h3D(jJTCg<s_2yCo||j{Y{PH?BPh11&X1o{OG{4WcuCh&wS-u4GZo>5wn#Z9%%G46R>OOI<zdrBwb!eIbTm-S>hM'
'yg?&!O@-Wc)lHNcqYInVVZ7M$+i0|8bB<hkrzl0a+qg71;DB|#;62%p2qp3Vf^A)rB@l4TD#L(+^75T7c+e}5XVYyK(D@oUA9Rf8>WH-#I^qo7'
'?-`Zq8Uq&}waG0RopkX4@3*@B@c!-JL^2h5!Jsf$U2D{rPj-MsmSmm5vBNR_1)e?2s}#=act}r37yjPkE3blyQ!ZS}u!(@s=HG=d=zp4A05Mqu'
'oUY>Cxw*iyyhdFq1vM(^71w`UEkr2GjZ|FC1f-%M?1G`)V|aifn~B3B-<86!VYF{@!MkaKWUv7dO`z7@U$H^s9!SP*qa_!D1mb+6*OH3>j$@tm'
'5<?ktl=_G?6hyi(%xmr;L(Y6sa!s{e#g&in@y1?Zi+#nv4^m?<%PSzdSaG{>E?P<;K;`Tn$L71>p;*dlKX=W&-l@+PU1{Bsr6B?+{oIj+pstDi'
'Ptj~JXixZN>K<<kaYJ4fLh(YQJm8C}PE7&Xdf=O{a2TpfEivKcnm9lhiGbWg44y(kyoVUWScq663tQB3+BKyrU(~owFL0`a$MZ6mSJX13Rb+)i'
'w#1c4(OZ2UYd|=Tv7zH?7}zF|2(b-Sp*r*iCo5!>!J(Qlz&W0Log890<Os;-4v4`W7$=|uef&8Pa_IpG_Xq0KbAU&l1G#4p=&<8Q3SC9*aNPC('
'1!#!5TPqnZAOIl(!~(t?#nzcP;MNADzCb4-ku`$Sa){}z_o)^$(g7%V{)wsfg1@BcN>3ifImet$Dy|sDzALThV(RJ_0cs>=3F#l#7bVCKNH1sF'
'ksC<uJilCffJKs)OtGJvG&uX_dw>FH*Ya|HKjfj$C@OS<)PJ^~OMJ%_Pne5op@CW2D!IqJIQ6bhXXt|~MG(n}5Cz1v3iQ%a3L_M3YLLciM4D&?'
'F_*76ChjfbBupj@#At#9fEqBf@WB!HHa{%uWms$#LOfd(-c@aW+VRzRg)u4M#A?N;*w9TaP$T072c3X7pE<^?O&N`XcJyptpWmq|MFp@?Rm}p~'
't3g#<U#AR)=MZme4WH7rw`nfr)^-iUHeZRH@qo)I2ym<U8dev>NR+|F8g^*4rE<(A)KHr^VpazW9Wtd@>c;>MvnA{ryyjmHO@>p4G}!{8EG+0k'
'GS?c0OfMt}=qmofIvwYvI2?<(<HLUUS$MvQTw>m`x{@|}A>C?7StuO#o!(WaFk4F=M!2hOp=)Yrr~>kXF%QGdd7a^$3SL7*fo@Yov!&_Tdruhh'
'Re+wT0oo$uxTDtyAi52i24)Sk2%U3*Dm?GL6v;7Gsj_rK@lyy-lmT#p;VHB&heC{3C!ha%a`N6xtX`~qha<KQ7a|&Mp&G$NW{}7bYizQDKmwKl'
'l+|r+5=TpTt&i^*WMz)tJy{F680Kz<*4#NwDalOYx~-yN3CojS@XD}oDOuRMLOiEnJR*H`oydI>Cle8=Ssi)myHK8CVp(KUhhG@<xTi!66LK)k'
'N;>l&l(*R`JrK4BXkNQ2GQ8LqL*%$u9%A#(%9aZN8n-q=Q6Ts=x~{ZaQ44T=*y?!6@Mn4EK&)0l#_UGN4=v&$ts&=|5Q#L#;^}!OI_yL(5MXKr'
'@R@jx8>uzCB|TFM@fF}|byx#&s}j8U@0WxvPE9LuNM=92VSVzXtt#zNS;~|Br|fCa4!=7+`}X*4@aE*~^iS`DkH=?cLs!}0u?|JLY7kr8hx2@L'
'?KSHP1rq2$GTQo$PN6dOx4f}sY>~&NBE2oLinhkWXLl98;UG9#Y2iJrA;icDDILsJQt=6i{eES*nc4&eds)cy0%9PRr~bW@@P)e&C{gOW$}X?E'
'MrIYY$2smcNQziqC@_TN?Pv+Kx{*A0I3UsElbS?ZRfMGP<Kzd9HGHJ~K}=%<6(vf#1*rHphGIkL@VEGsT+L$XY5m0^A7mT**%L{_RN>sg3fyMR'
'rJOdOwX1E}`sgN=qnVqNLVhe4wt57Z26wecO}qC#Y!7Q6xfRUtKZW8o%c;x`0Z?rZ*t5TV`b^l`Sg}Q>wpAjKLV$PU4IScQTytnd{Or-$SB|fh'
'927-GTa&rlb^|81SUNmvuvltbZ9A{g7L~ehoyCoo+>7=5u3v0jcLY}vajZ;dCf2B#FpM|^h~2omqT&1Ih;=%|)6V7Vx&8bc)BiINm&Q;)Qgi2;'
'2wHI^IM__hDQ;VA34s{RQN3Orzy8a~`!{r%4J(jjnhdL8Ef`%fpU;y;5(dVC=tOAYL9)sYFaSZLABVDXR)O#^P)K7Ge*CC~hXHKwQv?SOA7rtO'
'e!1n!5=F1kLrCleb9K_Pa{ROo-OCqgvVa%D10-Y{yRa4lP`H$l@h9I0QGFR~0V<5Qg{E-Vb93J0BSt6$;xQZrN0iwdq|GMJmIkjUxeh(r(lHDP'
'm9o{kbj`u!M|)?gs$`+s5%wlLNEp$>Mm&-uBc5Jwbc?IZFMuS70|&w%gbk`Z$Z|8Nm)nUq&F}HkfK@vxp%FgrbyOS=F)7jWT{ZkjCWFG5Xcm4f'
'^67dav;f*xw}+aC$KmHLpyWEnYR5c;6bT60H+BhjWBw!ql!R`JLnFEj5o=B)EQ;;7|B#GLDtWQIZBP9V+HaF9Jg<?(%fsdzgHM0yK9uq&NHfy&'
'+2uaMgSI)!LcLXoS?RC~Kd1QYXYMOH&(!($exmWeSbUnf=I8rK!Mi13f4g5M@a`0Le?}=w=cHEThQonqdB<xoU#3<*WnW%s)d1~_>-3&Osge=_'
'ez6&^2m_AhODe2AS>+{{WScv~Xtzk5oRHV(O>L)t#*|?lBbO;BS=8fd(?z6661{zJX71kGWLXJEeX~b#xjT3v;WE>GZkbFwy)QSuCAlwR%RE_$'
'<hlZu|83+071?9?jc(kYd!bAZS*9dG-wbgsJ9$c$nJ=w*)`?{)$%QD|vD3!J7&SCTJB3NYiPwk0M@+xo=vqM+ZCsx4fKtJu>=3j60Y4DF2u~5K'
'C;&nf)C$U1CILXA!i}qh^Lyzrq+f>N<s;jn5z<u_1=oSiGYZ$$toyPS@>l@iT5>mAKpL|xjN!~T(&83Q4s37hbGh<0yGqK^TFQrwxjD!+E+$-P'
'&$Ejn!M%Zt76s-;L1no<>1@f@EihU%rRE(^F6Q&BjDbd4>mf6J%ha-W6o)*gAaLKEzB&02pM3xM<o(&{hxcdL4c&rs=y@~^V8QbA-SMAJjDm9H'
'<$7K<sOOxN?}%B5ixSe}a-Ltb!zZDGSvVTPLKq5CCmy4Luj&9NY)<unT<^h}>8sS`BMqtTDa*JC$K^4VW`mTBY*T4%V{|ONO3Eu37KVU2eEy=%'
'u3!MKFgk2^FuOLzkb-Yl>D@HD#7u3-efqA{DQSu9z?IlVx7dD2h~swUmPh+8ito5CP;hs$2T&2k1cp4%_}wkTjAW>$kO`k;FT6aM(Y4mS!FSx+'
'4o?bem5EE$`efEMuH~%@=O9CZWLc{rpUWuUvB1H4hZw;jyI6anHQ&`L3WH^`G|{g%E=r+i7jR9o;q^1nNR+H$qY{p4??rwmQH-+YKJWKy8JU!i'
'sbujSPefF+k`R-3gD^f80N0WcyQEI2alFKBOEQ98bZ;`d)aT|N??2umg8CXE(k=7YHq`8`(mV9oLqQh}hRryaA5XqNd3zQge>{!<a`HD@Rd4bf'
'AU@nwc9&cv_x`p{zdwF;{I~rZTQ5tHB1`s#Rs@%#gooZ9zI}i8@$K>1pX1}#uTRd-cJ_ju+-BDq38wtU7LD|@HM0tvi`R5|_;8<?+&(p{j_zP6'
'S7+p}ifANE<ruYU$A*2`&OF1b{t-<E#sJiA+%fu=5UOgC&YUp=`no7Xz$95L95G||2vzZ#8ECr-c9oGoJ%ZdukSQQ98eq4_V&k`h7EAXvG5A@h'
'bMDGPmXkV<1-YT<$s@4P1vtSV{PvyNk+74aS@Q^8HLrNlV1vk|ijG@ELR#Bs-Zt-eN&bk>iwFSTyDb@Dpb;d;&wjrXv=f4+^;d}_Vj9&E_jhH>'
'49F}{%Ai1d5}&QOXIx>1S5>tthX<$>7Txp!6%RPP-+|aWjS^jB^?|JFo`UOIOyr9aqA0imi3&Kx=pj5_S66u<0+0h++;ij`6eNmdflUarR%zuM'
';)Y!zc^oo=qGlfueOW129^8CDYARIxDfQMNskdqg!}nER>7aX#+N!=6r6ZT9!ZHJ|#g;`_UFB8o_w*7-<kV{l#{@V?Tbr4>_B5heBuGo4$z=2#'
'bR#f*n<)z}6zz9{KL&&TOPIi}_jOYQGE?BB)TMLEdn!e-<#pKM9Hw)b|J07YVSZ%#I%tYw6X=1p;1L>+v_Pa55-?f0wp-m6k<td}3@HwQ3%?`Z'
'gp2#&!Dt%u@@bW03*}*KbiKS@!r;)F25Nt#J>$_YZMh0?=0-15SxihNr^01A{4^G$)E$R+O_2-b?CJW+Tr@cg8T^0{&>Va=h4D396T6f5kt*Ah'
'M-7s_3r*s_lLi=JPEPYxiYW>|m<O_1$W$z@4JI;ARa}TvV)n_%{2A#3k8pi*4l?$X^5I3DSH|cGrVk@xs>xN<I$s#SBKrst2(f$E$T1&&8i1S5'
'rt=dyp~RpX8w`DH(2$<@6`UW<2t_Bv#_*<8aKb%!pj$I^*#L4SN^b(h(_v`ahJcR1+nE>!Z4mr+DAI;FK_(^ezuCqE7{W0hA!3!^g!#*(Il17%'
'{O9N6rgmy=&o$6h))3mjoQIvfkloEpvM1QEK{-)^TQ6MlS}#Yoaa`=d6Z{iD4o#;vxH8g}$6XaS@&}^yr?|M7G|d6~Hof30NqYCCv{iBoAnRhY'
'KHz+B_GJz`XfwCDi#T(spN~4rdk&NOHM}(mwwQp|;@0x5Rv_6FJUnAF`rxAm{FkER`wr!2A?;!U_M+`Ol^KuGV{9e*O_a+z%SdpDnDYP*l$hOo'
'Mwx}L=Xjpzp1uhlG{h|r#4LBJQ%m^Y@7%JJ(u&R6Ol*mdxKs#!Thy^gFV{es8fcDcqk-U<lA_%W8*8X;NH>ikL0FV*5_h)~Ac+<`D3C1XAwg<9'
'2K}+btk~ST4vnt83cr=xhI#aM39jCB7)Ff(=eNcnY=ULV?B@KFaBf!nQYzLAwIp3lqx6;iS>;tV;oGU&W7Ok9NTAt?@OHP;tS47am~Fz#VQ`Fp'
'Of@*$veWH&6nOrKSeC|>Ag8%jEfe9zq!3|Usf$U``JaLKXKO1Mg=c3Y{oc{<x3(g}^E5W}{Mo)OQNW*V06_~ccMq$`F0cIUJU@I?Lx4(ILqn>a'
'=SDL!2?8~BCxT@3?&TL!tH>W$DMG1HM+PrpZ;^K6T}l*qb2$S_=6Pk93^1ON^86DwJ$;+pr<f?k?}btGrVy71@>*t=R+B3Mp6@y#694fl;nfc?'
'As@#-2jXT`Dh70k-r^eXvw^)*@^MPq#ojS-er+02i7UMrDpnlsUzt47l^-`LN;RhhK0Z4;`5eDJef8=1lOvgMKWtSb&2xm^ii80tQC>^fGi%&R'
'ERtn5qxIWyeX-pW1&y8->jhI}%!VXwNHDP*As~B)`N#S0^t~AC+3OFVPH?3C-T<vFq|8Ag#B!S6IRRbh%AG<gP@V}iRJ6l1jE8FK@zO<1JWY)y'
'Y8XLvcg=)AIqxlC&$`kjdDFp#=AgpjO2M|)z-`iYn&>!_&=Y*meu&J{6xj@r8+OQcc*K?Jo;|BeRucrHab#%i8W%HE#Y$YFM5b$sI@{HO>*~DD'
'QURy)>@s8ftnq{|i|o+JbX;J$OZL=KMjUI3I~#aDxS$M#sLZ!XQJ<usK3#WPd*OUMe0J`c0dJURl_7vFP;KOvL)y=}NgHNBPlQBdrAO_KTl#J5'
'yh93be#2uj-U>@@gGe4U<s6uaiF}}aI_w`#H?c~s&<dV{xR%V)Sekbj@NloDeA}Ywgkw#EA#kg=SRJ(#H?%Mh8Dfj{9xVpG)%(ZlvejU@)eJ$y'
'IU!c1%KPQSnpLa*c_-5I=<l0MrBoT)#K?Jv2dg#`|G3P4Y&7k!Uv0NYtfT7<yxf-QXVAk0z5ado63vrJM~+t0oe$nae>{CFFP=#~<S$B5E9iX>'
'{bAb1JfZ9?qPlF$Q^X~DAlhcLd!Lo}7OmE*$S=~EE=}88SIFF}yu|e~&B4<5@Dg+ljc2Xbs>>q%2Pc0(y71s3y1ywfUx{Hlj6TsVo8B82<lgYF'
'bpYPyK>WzCRYmY&<A*I^d=)U)2fB!bEb=$(J)3P@EUTNs>C1m9KnNmhS+=^wR@Wzgv}8K9O_9f%v|9!Xb&3gM1#mGT_Oif>_5sczNsinZ4oMSG'
'+>)_8L{b7XvVqi`4zEKsMp7{YK_d}+!LDwZ{F+(`Y>Us}Bd4%Dns*}e&c`if8<?>MY2;=86k@?A0Z^a-=Z%hg<h-CrNOvP#iIH53k+mWt{$oT9'
'a+<KB35l;VT1rz<`92ErRWivccZ}j`#5g{35EZ4Iq^tqL4;|j+!y2{o=$BJMisnj%R7a^NPZU5NPFc6kG||aZo=rwcv%<!sJIbqQy69CT0(%AC'
'_2y7tL2b!>UP0kynlCWNBsF^n|K4HldiaYfOuq6T%FP-wQosZL;%(3jvf6|)o@yd&Sgn!_C+4YQ#6EUw8L`Ja6`hbc?o3&+$~@JbNYZY7`?b@+'
'<8@gzz7evR(ZC?c0Y)-3n`s@Qzl*Kv;*Ak0^!GZjKu0lRUt0x)B8K63=yI$j&Mw~P+~VxwAOq|@2?l*_J8}h7l)_amC|$$NBA>RYi70r6F-IiF'
'0S>pWQnzG*fsbG4QqTu_SYZm((kV4ig`_u6_&}X+h7Y1~lAni|8IZsB9=Ij?5jr!GmXl;9Tn~Jn%BI7CGEE78_yHJ3fqy#s;R8Q=FrRJs*~WbK'
'1F1!W(9|i)h|T0Vp@9*^d>HTr`QZyriHwA!ln;*_h{G`aE3_s<m4L{wLW4ro6^md@$7g2&H4$K-zY0YNvLu~nE6yT|E^eHOgPX)Sa!c23iF3g8'
'NPnghHgdBg@^iWc0DAaOo9a7^iNrrRsV^^MGLe*t?D#owm#h2L6{cG@Q#;G%>F;=`=HeWsnlW`V9#RJ~`mS`(34eGmU^kZ?E3=~&1%4y1h9%h~'
')>cL;;@V2G(dfc@IJ$9{|D5a9P<h`duLDkgvIL>HI7r}Fx+F!UmAx|2OSiU`L5Ib%92!`~8?>Jvp*r_D{PWUeIc5NMDD?+;fA|9a`Mp(0#M_q5'
'`~gcmw@MiC+C?+a7yfINl8Ryp&hHVPLs?{FEs7csiA_dqL-k@T;X3zjmgKT*PjIUoFYoKSfFf+nQG`!K785zIignfZ1-5C<Ig7V?rA8i%z-3a<'
'3rxj!P=hi9B9SZ~R0Bv5@EKHYKiqAihx<(@JlEH-(3A-saB&&yUWT$aFIE#HG{p};K*{#tupbRx^gEI>P*w}ORnmIP!1)cWA&8g?P{0^G4gS!F'
'FAQ+hj}Bk%*#*Vp-?NkM<2za9^&tr5+Zbo0oZP*9j<ZMK@8$Ojc-r7)KjIR=Q}{J97f~3d65hjXc2sT$fwe&A(bG;)6YLSQNro0@ktjkz%$iVC'
'x|<ohN5-Zf2H~H$i8u9V^)gR^E_5ss7Z_z^1_NE_uQI2BQHqlr=raGo$qr1qP$kCu*O(tdY-QTC6Uu%^@w{D#%^kZ)uIbi89PM^Tm6xHaL?H}O'
'cgBxo37Lvfh%e%)|K3E@NHG+bs6-xo0=1@D6?4TC_=+`Z>EdbW3^l+4QFyvj-;L2qQ#A?#)O<YwxrI`^DL+=uz01iaF$TGu2E8JpNx6r(2~$zb'
'Z9W~z$~_(D7<QQRkj@yoYg`x=TF&T{Bce*x0=^Z8XF9kDC9)w3qt>fd7|$`{WD6Un-y4`!jDbgDj0nh&*{ZE$#EG$cQfY9jU)Kh0L~j=XpzE{Z'
'b+N$Z)dmRifF=NYDq8bTr#iCXO^!s<NY`y%2)rrB`4UG?J$3Yu-NKc%%J%$&LPR9I^Ib>NC7_Lp1<5pfcz{ZS6tM|xy_v|lhEp!m<nZL3!<fa('
'ad&V&ra3wfz6M12{c{xG(Dh9epu2$I%ZEzpyc_dGoD&q3drelL8b^mNX3#9iWt6Rt`{xR%$OBUp?A4>InG*xIFT2E32?9=tVRI$sTo?Zrz^emz'
'j14kBJ|mNj;B-05&k@u3H>NKd1~*5&e*Y;x1=@K3DJ{Prl0$*<<Ztlg4~O&w4<Bd95{AJcx0KA&=_>#$EsmGdv#SJTf?;r<vYDI7Y{uUmJq@UC'
'NkDu<#*o9{J^HZ*-vH!G^$nHzQSh<IFdF6Sd=A?v3eM<oYCQO`u5eD}&sWLhT7AzkYtAX0QL{-Dd<zm_fkmMa*kIQw)P%1H`}gAS-{3E*v@ln}'
'F(zljHgHsD%m77SP>O~Kg{06I1@DsLGF!ftUp~n%-$lWR@RPS@LQJ6GrO})8AIW#?;0y%X=;&qizyI9te?uSl>wx|TU;ohOPc#5Q3ja6g^MCX`'
'dItX;Mj+v!kn?-^&)55(^ZeTC`o+=D(Df1g_fmF!)X?<@j3#XL{ru<ZTMheVL*KR>^UDD7Tmo^p9SD}}>_vD%y>Zv>*L`7l{s<U^;Al?-BjM4}'
'@v-RH`1(r#L6WHr!iz0y@r*w3_4VrkeIf~jZ!O?3`-j~ifbeTCME_?%hUX?mJ|_TNqIg0nb}WKO_asQENZdF<{l6eWOe(jLo7s%=34O*m-J&0Z'
'x4X#|lupEE^C-i7&4oR|8Vrs%3*U%%*TbM6Js8?Y5_mS|BU@fd(FS7^$!#cRPm~w>8-%c<w4j*rCYyjTukscMH(e#+_v9P8xL1$sadvf(4bdcC'
'%aGm$Pds$3OO_F9wlr`%-slN!D&mo@mBEia5#l~i<{9tQC2g#z<?8<P<`!sDJGG@$2~x!MITodyL3LwpxyH^dq`m5PaT5myyO01O#0Rw_G(43!'
'P|v1N(+PDp54KWO4#{$R(6X3?3xo$0)gN(2CCj2moLo`IEHUw8*f=5P6g-dm!#@5Wq^f50AM}2Y=ubRY;r@fAnb^2vw87OIWBd4Y1~48VNAmZN'
'kTY`$Ab|o5di`yPtV(u+!`@M6&*|BYAhJz(%2I+mJ4DADr$>)g&yS%Q=usSSE{~demaAn1I#l$K`o0$2$WV4<WlT0SZh$8Y1!1dxe8F&F%4@-V'
'w#UtJL;3{&6&!#2=JbOV<wwUaqiEqV$Fadyl>VB|do>GDoq?e#N;ZT0`oo)(*JP)@uJU@6R)vVVJ<@Jo9lkh<G3^5S!-iMs%|!p2XBT(y+ERH+'
'eZFiZ3fkjyyIu6JlT`&Z54)Tcg|7MMRgugRPHx)8QeAfXU?uBpG#FlRkiSuXSWLQe*iQ+%o4`-BTe2Q3E~+9;M|AM%R%^~q`eRz;sCQmY?v?Ve'
'G0N??@ZbCIPTs%fyry%sE-;bdAtaH2918I+7miKle9UHxRf<-!<<+{%=Dl0M_N#Wtk3*Z%^TWMt;8ftGa(%HXa?G&NE7r^QnA^b7UZHyTB3Uju'
'AZJ%#6s5c_t0{WF7?nSsemtS)X;GNZ-<*8+{@b^2wZ{y`v81$0*2RzcfstI8cq4pc0Nx(=yFGT8jdB)htd3|TlYio$!)KwB1k}CZ-6xos!o=wR'
'Di)eyer9)MmY}1V!B!OtPdAPk(k}68C>68PEPJO3_SrYzZ!^FaR(y5aUNLjp##30Bvni&q<F(o#19MLVyIM<}?`af_$-2{wW=2iuP-fV&V_8W2'
'(oe8ma;qHZ%t9)(mVBsheF&^N-1K<dw{J3eFlsfG;p7Z3pu~$TT;R4Du}S$1L_SOOhIOWRj5f*t=F`c^*@ti6zwx_BTPs4<x2?pEkVIxpzpJ3V'
'_mtzHnucCpT%sG{&7t%iM4o=&hGE<0NBEiM;DN@nspXKac?t>_D?k^%Ne$H%#ke1Q{t!Gn=Rlq5e2Qi&VK$9OodrbUCQEP2NLZIe=r%`h&|MA5'
'61uCQz6)wScfoKy4UN~G`YdjBI20~YO|g9~S3`@junP&ta<ZYTtvl7I$oNClPza76*HKZ<jTKXapLMVP*<Gxjx&C?EYRpBA8t`@BMlaxooM?Ew'
'z<vwi?<~|?&~;7hVYj*+#}Jx45skLl$m8O^0^<L?PnI@^N8*ns!HE{4G4+h__2z4ZDccr*BLVvg$Cc2Lo={B(QY2cqz5udth2*_$j6jsNV}NwR'
'PHlG^Ly+Wlmq}E83<}l&Iox_By!+USmd0k8K@DCSW>B?-(@gJviCM2;55+p8&`)x}Y9^+}oeLNA+`MS_D}P2kPHHflc^uQEYQo5jB-9*qzR{0u'
'ifkrQg2`>@_BW#2hB8P(4Tvld#wuALP8F$UM*bG0c`F?eZ}Rzik$Ot<3xiAAqiCL(Rkk9dVp}4nrxH<Q+Ml+sTXY|36ly+yJ~MLy`MDoMAap5i'
'dZ1cP5uId_jr~-{yR~cy7sCfb0CEK9lfKA-f$MDL_5Gl3q$wGseKXjIrS82Sdy(lpo0gM}UK;e<F~u-1*_?%{g6a3tyG?EkO6d(9;6ItrRG|Re'
'<=h4l&pbU52{}=cqbum5ZNStyM~f|ZKeEwO#(I@U{Kp(?__^fGc-64wP(xIJ&M&-J6R{Kf5`|3tgc&f{d~UnBApLM7htllT{ap0X3C7TyrskNV'
'AjLA}C=}L<VQ1XMn55^<5CmGOt4oWU>1h?(Wp|Uq8mUKCoaisLsnuIb68=(0u-|S&D@J1Qj*E3RpT^00n&mO!$$tmNvHap)oZblI5@Hv48l)$V'
'g5OQz>SIXwsjGs8<D5QcnoZe0p1DfZD_&q7@iJmS*%z}Wuu&%_kFs4!l_cgsD`9ImHR&FC!Sqt?0fESNDU!wdUJ-G?LjetRM=(;Bfn+FBCBAsa'
'qhGfLb!|ZhEp#~4LkgY-Eu^lkjybha(0lgW9v!JI9SXIycK3=<jT=EU<8Nv&qd;tXl-4}T>RVQ2)}l<p6#jD>JZL~SUkS`DuQN0|QLKDjW|KzU'
'<hyUrPG83#k3av}cUYymvCMZrvRvVv<-~x$&MWR2=yVJAMa#3`89IPaLT9IJO&iMZDNKt<rmo={*0z>(CJm`vSoaUDwF+aWBgl4FF6UU{X~pC2'
'=S2;Y?CkmqI~eGngUY|7NtHnFB=BTH$_s&$OzI^1+0jgO+w40dHR?ZRzy!tqG((^_AdaRUt*u}_nvkeV8kDIP?b0xf>z1}59o-Ns-eF~M?KDo5'
'k2;|T7!e~lRMJ3N`m%_ovuQR(X*rvsE9Gy$?U)yd6g&=Px7j~8XhwU<(n*T>-!YgoX=LN{ZdJ79FUf>C2P$cIgyIczzhCpj`F(7{Fnoow1tx(O'
'Cg|UML+X6&0mMjL6iOqTp%MIw)VXkc3nmH5+}xS%AdoF9BYLb}l6KpPCPBx0lurRB6Vi?>0K^pi7^$EbS4SkJk@kOT3QVPwKl;2%kKi1X2;llr'
'F6@Wu{#{N42DE$N3S|5?I3M<&&*W_``EaB&SXhb-!*>D{zh~N}!!cTta_vZTJ05@-Yvn_!@H~ocEqzKPGpmeO1MzBr1en9R$am$6yko=+_n~Ap'
'z*+5EdqS?D983FwyP{D%6~C~a@woG(U3DKS@#EkDd*bh*Nj|MP%4Y3()vlt4YQROK2dnc<^q^~~hLp-#zOhmsA-cskS-A#|#`?`RKP0<$(@RNu'
'oQfZ_75gdGG2ixz5qPf<8@oW+5(pBiqtGi!WWP%l+A-SpB=Y~5V@56*0?r?8loGA>SZ{Co*QCkhN{IRF>J;2wp_4ymM260Uej0idWu~x8M4}bg'
'B0#8nn7nmzEz+!GTIwR5bE;{Pg<7O~M~|wW8nFo|xkxuq7M_Hj+FcRSOBhK@8FFv|VGW`KCRZ3SdDhC82M;jsn{LZ^M<_wB;hQWGNlE1Hk?`F8'
'g0S2_5|;a45Eij3K-PfABYsC~F?t}B-wYpw@9YpT3)mzGeV1_V#V|#ybW+8{(MH*Ff_}|u-Ed`6<f~d+kwE3qkE7NDe&4i)ccWH2yX|JT2lxQ!'
'Lht!g^qc8)TEkka_gMI<n_XFj@2T)~G#Hk$%R3Fe9G^605PP4JK69Q!5-sifL49+$#iu1J$+;h|JlL1Xcv;?NoOhADg~++gd`;&7dc6RqePzC9'
'{}XXK)B(28j5jv2GYTG9%J81r?ZcnNTn?k@d@mU77B`bE!J|Zp-R>`jwm>L*6F5yQ1yvQT)U^nYaA=`Xz%BslH8o@<Zp`og;!jYvkNtt*_BHzh'
'UqTVlppi~QwxZJ4k38Zh7jZ;c!I7Uw`}@Llnp|-7P*f*>M8}=Mj`0{s!U-!;W0GV`v^bOjLvSRj*=<H!=2?&qmVAvyLFL>trxtsyq=tsvpjsah'
'a|Iq4t7opzEM3b*7EJC!H7(F{B>r|xU9+X~<@fP}#Aj0ZE>=f6Ufk~DP$-mYd?HK}5S|GwWMt>Y6V{ev61OGKNo)ls+E+FmzCPWO81F<KUgYel'
'X|-3uQH*FeIWf!H^GVwUax2t{YPF*rwe!UySx%imbJ}Ky;u_a1cjo!!9Ass>YbJd!J!K&*X0huouC6fA8kwgYhRri9;;|2;?F^@_pi>pknMl#-'
'ZYwH4*8u1ar)B!>&Ty0!l{c{pwX{qMo+~%pl9(KHjT^VRvsToaCpWVU{&wM?S!<Hd^I|j^_3sA#!NuVO*`$hP!WRgVYdVi56D@QFWTn7sAL@js'
'>mxhE-iz6$Wn6@NbY>>}VYe*Fli2M*wDxSaVPha~p=R=@Yi8o2gx_||0Di_=0W1T9?cr0*|7$6xEgs0u@vcsq;{atcf<;~B#j*VGWzQRU%9xur'
'P7dWpkLP*5qQMNTAyHj28nkUSJIY#6R`k^~dbP%^^c{0KPBl#(S2<FfJyg5axv_`ro7WI1C^uhDG^jhgl!tXJsC}2gA$8bI&R&gCB-&FrA-M^U'
'YgKDE*{t(?)DUWoHVh_s>YFw}dGNRHW=vWOQ+;8<fnlyuQSt1YgW$==TIV}h3Y{;_zUpR8x5<s6ZsW2<;T>l|<BnD6QBk?BR9|F)-SJ$yijG|?'
'eTo%_eRHTdg~4!UDlS?SFOI5vt_itWLz=pgq5hV9?rL`q)FXOA(A4iWlj^C{*4K)cJq5{ZzbnetpG4a3X`rY(X^m5u!wgnWMpFvPd0u@5vMB1m'
'mV6W^M@Bb}nz5Xdgp|%@Y%VAXnBF%I7Kk$Jrl1u3jZ<m_%RQL23{((z;G5RAnj(m<FKn*)p)(r{`$Fyt49KFM#gnV`@;aR|iuEIVeIj)smn<*W'
'$z>Yp?wdm~)_QzIm_^^(f}}9#l;#(y*|^9oDAEkd_3?#bG`-pPh0a(GNzC!gL(*8pCM)=L+@kNT^QW*qF&Y~`gm3i7aj!8Nb-Mj+ps%S(l~g5*'
'RT(666twO!Fdqh7yoB9FucKhSoM+ccxpaK?spkj?0vCl$PdIE)@AV3UosAw^*1@SYG@~oDO<`cvkF>xrQtJ70*Y*5yqBL5!t$0H0W-jvSy`qb%'
'j4KP<o?AU}zs4~R8Cat0IwI{uYq73Ukn)7ir?s(NyyWMRg1tG}mVhsG?|L*(4HU4#&)*Jj6)vgt=z=NL#i)k+eZN4Z*@xsFBp*9m&+=E=AwZp*'
'd3(eyjGRd1!(&%Ysxy3BO!9;Hc)5vetne5rGKeH4V}O;9+LoPX<XwWozbPaub6raY4wL12J}-;OXmW)|ra6YdCMzH~9i{oOOz?b!ioSGo^s?XY'
'n_dGY-pCSz=7Wtfh+QLdBs8}rY2Qc-PA9d7@Zwe+NK(ifvYmOXiQK2MigBdjCW(IV!ET;h0HFS!sTi;QwqoLTft2}W%o;qpjW<q#S!5mrwfLim'
'f@y6QJUaA=1_M<g3=f(}qyD~aZ0*dA5ldANuUx&fL+Y)o6O?(kT&YB!C39m<_@_)Gj|YK#V9dfVP-ah4q0C+-(|XNzj!Npz!6AG@4K#k^gY^Xv'
'-|I;0nGLy@@mF8$j~}wl@aFu%f_D>P?3bh%4e-mVDn}nbpLNeZAAkA`U&c4Eyio1jw0*20s?mk7{G-$fHK-M#D3}loCv!n0qNFOah2o&2hbf)+'
'qr^9m!~+<cF6<(5&2yQY!x4!o6~h)KoM_A;YNZ5NSmi3k0dM2WBm?SrgpkAH{|uw5VW<aui}KM>>fJtjOt&?<Wc22i*Vi9D(qvqY;12Qln;Z4V'
'Lm5od(tiPA8Yrd0Bw5CHcu5jiDbVc-Uz@v}(j>rZ7nBYnZ>=M{UNWDz4Ql`;;89SXJ{76m|4>Ojh?-~VTKC+OR%P3u0Bok-+}NTp@Y=4mJVcCz'
'lV#1dRAo6}oSaXiv)__C#E+@G;=RoAhM5d7&?XUPLEboLOAs6I2l9bd3CLNL;>x(}nGJr+f;md^bHF96V%z{@;>D}149NMfxWQz(H<;v`iyR0U'
'-)v8v29%g79Iu6wG@9{DB0Av10D=T1@E=oYyBbtUHaFmt>q_bdbTa2PuX={T4L_0#G|tXF3QO}b-<5RnZpD9=;@#}D`gG9HuJJ;*>^aqOt0WLO'
'l0Xds?TjOc>KL)2Pqj%9R@35=P#|qn+U4#EK}aHudsgo?lx@g+nFW8OozU2~4TD1SS!;X)kB~BX6a)mDLpWOxRCiM$moBS(72`>8G?}A1k$xKl'
'R6+v_OtkdD?7kL*-iA^3XjXY-RspLWUjrhBm6{H&VUY#yBUzQH4n3#vtM5m<zJqk0)n;v+My~Bhhrv)igB0?#xtLT_sC;}&T3=3U4&(}rsfF_m'
'(YR3)YP2+*qlX$r*T0#rZ-!pXz>c~D#*blRPaNcK^dOdkvB@027n|JmH7oc~iI@3lZJl{G{3OPD<y&w3c-g%zR^!L!R~Gb-k#1xR(K_{j^Yjb@'
'qEA3VgJrOoCrHMl!7v&1$2VB<+|8nB0~g9tjpDWig~>{HLeU2(2IHg#h0UTvS;Bj+{l%x`7a#4;`EHyJB^>Rg&N5V4f!1X;PvtCpFnZnyiM~I^'
'CZ;>qtYwsHDK&#}Jt6M6<OLY)R_9{Ex*BunBEK_k7qp#*9Rr6i{4gRA2(QY3m7fYLFN@1N4U>IF?T(@Du_9}Ve0^p#D5ox46dDI-1nQ8(<I_nv'
'cp{pjU!4v{rPoD7`%kRyu+aYJ?VqW`9mn*4?}u9n*%!=1_i+1(-#{fg`%}Z4NE}m(%Znr;0L6iIk=>=mC`m?KQ6?MVqo8v@w9$f#_zsKtjpWGY'
'&ZsbBcNL=;z~Amkk>?KduH&8Zd!waVcq9{(b+I+zC#7Dn_JWM>#dT^)Sd992WGWd2_w*OFVO7zVqv%BD&wJ1Q*&|XoepKs*#W6un^V^9YWEY#K'
'S5Ml5E*dgx6NApdgWmK_L#nl8Sv}9U0_?^ee;ml&{~I8?nNjsx?gsg8^Yr%T0r~DRAlI_6J`U#YzZPba7PlaUfwF_cCnapZ!zUk4pSbWHnw@#N'
'+7-3mAb{w|fL-ff$}$;fqCt=Ff<pfW&Tw0k`|32+E>A~<YLqSE_y@ua@)8(f!wg3bFXb5@CiB%5N<n@~xdnoAO>dtXlX>v@6_i1B$8^629}H=0'
'JQ%`+ZO+{mSPgSOx*N3~7#BCK;XVAgmp{^c7AF&84j8a$$o2v>Eqv8KYF(t&ZHjue$N)EbXp&DDHQ-<ek9;Gsg+x?8n+2tWC1XRHsz#JUJ@G5K'
'0sy|APveVqL06Pe9R9Vv*$g!z2B2$O*PJY9k_IX*%lINGbl+CmM-cs*YRijI_9Av`wRc-mcwbU)Y%A#jqj5rHul{g{ZK~`<NmPxx;Q>1}C8jgo'
'rdbxuo^-|q_6k2cPPU-%z{~+O<h(#b^I40ZwYKg6a|3Lk8$fH>_Xf~U%;ok2iYg@W4umOj3PP=yaTE{=mg#zO9ZN^sEoM;3^1f|aQHfh(pV1Ya'
'3j~mSKVAZuAlzlFr$i+QKE|h{_7d7R*^WMGSS?0_&Q4P<^Bfa5-9oEY)4R^L{@jy|+Tc*k*x$0iG=+U1c+zSm6=&4yyWv^oMU-$O-F;tTK<>Nv'
'kvBX1S6rRAH!7B>oCwZ=k@bK^GJB3>nQrf;U=Gt!zxUF64I^}skH7o*lCjVf!x-?-=D)<>?L+jyM|S=S>t#GQ2cdQhK<Z!<%8@lXWQ>N*CK1C#'
'aiI!yn7XMmbSk&@0+zSnu9>zp1qxhw-C#njcBzlc*|cQbKWE52?KlcV)hbu0b>b9PW>uPIc+{@WC{@{~trqKm+%`H-o79>KfIecI8P*Kds}459'
'C7xC1*^eoTR}#}O!x@mvaWbD%rKXkkOhc{8u2mGeVb#n=eM8eMWbq?3;|Df2pN@L{!{<X*F}xfmRRzcX(P4jx$!lQ`j}D*r$B0x->ug}AS<HY@'
'V}eh&FYW|&q1vtB4lH$O!8~72aRElBqwe5n`1vFJ={+C5{d7DU48{o%uX7y;U>_Bv2^p#eYdByOFm&d0NjRJq`3kx;l<lK_^o1actKl4$GU7k+'
'Y|iP$YSDbv+)_EB03raXNdxxrkHHUzE46}aw2a^y-(w)vkDpG?PCmDy7T#%MKvR#Yt!PIX<f-7NP_10A(n5S!O++pH0~D6Oz#lYzSTC^rJil$Z'
'$0_{yY)em<H?SeC;8`Zn?lqG&x8s%w*V>BsZ`A4vCl~<Z$aK~)E3$XVq?olLlmwNKTg>E@q+^R_x!Itg_?19FftQ#RsJlqe4$qL}^uthK3Pr9R'
'-h`1vl_v&R_;LmKI$Kb{+57d)(P5uW`wC&Thh|3HMnYz93B_RUUeHh2d@9loZNiJJLTPAoO@$J28DjIdtrm5Q{D$DcA5U><Wd<pPfUkYzG{qRz'
'OUxBTL~Q?|FGMD#;eCP;FJX7S0J#Z!JU$H!-%sh57D)fGjpLBW9a<r^#G~!ew0wSk!h;>oQa}#Tm^l}1>G)D2aG9A!T?|>4s5#FHz{8;&i=C0t'
'$x$p?vm9(O76yfq#Pn6Z!I`C%58^Ojv6O)x>lf!y@XYE!CKZ+?`S-b5MCBQlWyAxk5Fx8khH;&rS_OrV?<`F)kv$H}$?hE1j?qx2_RO(c2>mv7'
'P}Bdytu9Q}*ni2aj-0}}3~|#zhrUZksOWoK2YOYjxA!@HmlK~p6hHp>&L?*lTxCm#YNi_C9DK%Oc&Ee8s0%r_Wp)*^no<o>C@gFuj>P|cw$QaE'
'wLei~%lahn1?{tsQ&ZE%(fLu$K6zp-z42BC4&E5v{p%Hldt$m^<SS4{zi}p`*_a;a&F<jdu+Bv4Vjw7Jb()m?+mOG|ec<9SzDnm9>{3)hMJzl7'
'>YrSug&7h00>n*{N|5BqFg3yvIec~e`Y*5E9{=Y_{Oa4+e>wRazdwF=!m+=Qyoa9PzucZb{;%{Xrm=~T*ldR>RxYNeWGxI6wnn{el68eA#PC5+'
'<7QanLNs{vx<)$;)<ARHuMSZ{`R4Td&4GwMZTLvzN6ZS;v8T!Oz!HyoVvQ+^Icqz3z>)47<1Zq)Jy=i#{DEUE*)Y`~zkT)g^fd(oe|3C@lWfFb'
'tG}dZsRhShLuFV!M(IOHgGJhhf1wJDOaX}@xm|!UDj9|9en2=k2L#%I+*5M$$X%HRLUaw)fnr&AM;?QND^8;DIBbWn4g;Q08Ha8d{0^lVj5p>P'
'keGa(jaA1cd)Duti@lhjJzqcpEY?dadc~GNlmEysYSAkK_VF18ui@;7)Og(M_4u8e!SqXf<hh`V)()`pliI+O+axm!XrEMoT0$x23s}W<Y=-#v'
'NAmmYhfjZr-<*CzOk#opO_hK~!2wMw{2GVw4{SA+bVD^Q9Ouwc6o?}E&Eb;O-RL|)H4H;Q9u`B7scA0LQOH&r0F~6DV_0JurdwEnjzrC5LY`|P'
'Or@R_1*eQlh~v|>5IeHGSLMP9zimj6k?2EqD6g2arH;Fa&-BHP&N7AdP0=pslD&kARL8lc@F?t-)J`EOhR)=sk-pYS8o?}#JX1Sl>9ceW@&Skv'
'V*YM{@#VL{RbmFsaN}k87Dsyi(dM?-XmBW}PFq5H#%cM(iSV_CQ(UTolWqy>VqlQXAEmQf(>m9kHf86Lk*Ez_Tlh!Kce93BUlNi1vpI>uda1%p'
'e<d7B4yugIq-Q{ME-*0Gx^&|(9vN{MeG4VTXJA?L`!Xx-r8<1cVKZ2!8KFdN-3+ls_Lg}}?@)TLVi_|^&Y?AVYO12(=%tun*#TfSM+yV!tLETC'
'55^(hL5An#+>0kCHchnNb!6gTJVSB)pBBX>MJ9;85#s7elhT7AlrJfVL+Ghsrn=Ij!P33yda){<6dTSd3qMSVG1D=@<|!by3U6RMs;N}UcB*9f'
'*G+6e_4Trgqm=hxp^D;Zko_z@1&WmPILnGcwFzWE>4CArQ^Qz{BB$Wt!$1afCBIsnr#2Z@hcwGBVD6YuNeUQz=sZrBB^+hrq?}|KM+i5wco^T^'
'v}1NqMQdZ}N@9!E2V*pB?6zk|k%QaDzN^Q3tTj8@YlgPa0^rnqI<%?45yr0cgCiQ_ij+|OEviP|h~IJ#$vsEDeE9DY<abH$wvZ3mZjWo48&QB%'
'I^u&2Na}VXiv2ctJx>-Z4p84!DkGq?mq5ov3_<96E`3CVV+Y7Y={#Sh!ntFDjigiM$}PB$;m+NZxw{@e%hnE)myh8*Y37GC->J0g^)%{rGfUA`'
'Y?{FdDu6-;6Jxgfsr6x=7GyhGK1rp9(C*@?r!`7|X^quv?rR%JcSz2BOBUJaKW{Z=$qh!Q$hH&M8Y2Reze8D!kjszgYsWxdd*#pq-2u!}a&BE~'
'5h_4(ZYT7Yhvm$&s@fanigb}=OHLPNN~FThN{OVX`EXIL>p`^bDBS5h9?>;~eq-SaSdZ&%sONy~L0v#Y)k`zx`G)(o4<cs)%*^t>U=N{&4VkX)'
'apNYi*Hh6Q?E-BJE;{g0?#hN}eZtMd3EKBFAObf}S3fZ&22eqo@rO7;Z>5c#PCzCc^AJZt75kbZN>b>IYA0C~nG1Y5fYlForo=%lGP*XGlbh5?'
'>Qs-^5Nv$Z)w&i`UOY5scQ+CWq0eYliGHfrjYZ?^6~GWgMu+AC8_Av<m(!DU`R89qow4|5TuRyKqZiYx*mYprvY>LN_Fht}WwoeA@Ak49t7Uhn'
'c$N|$NsX<`*Yl6K4UW&<M#}YSHBXT!B*ndmw%~0FElNysVNXphWz!k7n-`P_lN(jJ2{|Pb(9-+{Q#jp|D3?;sb&L<Kq-6=AsSEbweSK6q0aMeY'
'*onF7(=xkjtf~{j>`;73K&v9dklVJf{`@Es8ymADW@$;{x@UkAIb?j%MY7ChBz$nWJ;ZZ)bmpb2#olyPAg+vV_p{|(!fIty<61*~@iLO9cZohD'
'b?K+ZXXo8&RuIXMrwDs#>UZl^<lTg<A`&0&R|!d^ydK9``FQkb4{9-BIArHj+saXPxIKWZD_xlJlTD0s9tlKnHlXKBFGA4T4}FRVCpfJ?K_)^X'
'W{})_*aNF+G_@Is_8G0h621@C$5??Xe#i_lUI5t3uie|Kx)Qan&X`dwVoIinaI@?$vMUntV}@j6hPtvE2BELP2NNoN;Ey?N=*FqjsiP2~IbvvA'
'h7LX_&frXMlQN;+px6YpCLcm<2lY2zh^<yP^=5g6&>Ww>IE;fQ>ubw;&3Osr_Rtp<&~1NQFM}WUH1P3Z<+`Ebt9yJ>Sgt5BU__f;cR~^D)G6qO'
'c2i4T57JsMSaZiWCubBq`xlI#Jbap5*_Qb-4S&&L8-5#2+{z^hZoqnnW(ZaFZYsAofwiZ64??RI<KBkc(t?pT^y;IS9d`mXA>MBmCAsF>PR}1X'
'2OTfW*fg3tch$9Zx}&7*y(ZhZmUFg(Y;d%jb6D0*BMG=n!ED<;HD-I3qx~nyrdf3_CX}xy%qjkfgmo++`jUp#$b^*T7^7@mHhtr(4J_!ngv@jY'
'vQtyaCe7pWV}w1@DjpsNPZVi8pp#yM?9`DbzH%h~>Kjp6zDQ@;l1?cx9Jge!p06#FL-#6P*U`z-@rg-FOrse)ioce!1yE6708x{<>rMNmyODoi'
'z(S4i-rs&IC^-M|QoT_#@f-Pe84<6MkA$(3e&oIBQRHpZMCfSaHj1hvFCmpPJW^9_zju(LfrySg)YL#+_bB68LuEhy+u)N%nMHCRTqQSnP$lQH'
'sR&7>_?N)#9#o?6+~P)Jl4h$)wx@H4BW_}W+iXb+M3U;nr9}zCDZDE%k=atDAEy9W6^0O)7RgLCNY#WnSPHC{Lv+oP5-||Y*~rJ~a<%6<D>Wa8'
'{{K_Y|G-nv|C9rQaAT3Q9pzkUDQQoG<iwQSO~p7W#}Xa8!Qh;kyu-8W?i6?&w3_n>)^2y6^B{iqldr7)x5--Ohi$n$quefonV-GoG4tccLi5us'
'r<tGqWj4EaTYhU^=54Z^`SBc0I=j(be*6pI5eqGZ@$dWHBB0ZVAi5OO<`>8|B<)nWgI%=D0mIM}?xIjuH<pyE5=;1Zu5}S1*`d66@YWD#N&7KU'
'C%qz)PSr-ehfRv~Cn>c<i5!Vp$~RKgqy1PLl{5B}ga805d)1FNAMwdj_WXMUs%_#Bt8cf7TVH6QBC9eDaIj~Oy7n55?4RZUiYl8W6Z9&yw_Rvi'
'Yf0N-^I6?z!KGB@LeuY>;{;!;HW_kB1E!9(zFvDX<8%;<a;E8-S=F>WGwjV|t){uv=2ISvDa>l_SVwKq<;!HE=IL&L*K_C_wJxm!uBgs9yc1Y+'
'z$11A)DJq5%4a%_lWG(%^V@buTvkl8f-2~n3mD1&3s6e~1QY-O00;m80AgKY6}YYK4gdfSH~;_;0001CX<uV+ZE$aMWnXD>E^v9p8~tzF#`Sjt'
'{vWO!21zZ&SW4VwLYXxS+M#GybZyXI41+n5Cy6yhviz{EW%=Lt-hJ`zNJ?_nU`A_`$9M1U-uK7F^SobKRi{PH5}#BoPgupOdCuCVq8VXH+K{p;'
'wiVrzO;M58-;p=PzAS1MTwYvWyxlR9u`P`c<lQ^*A#DKJPZ`bMy#suEn&vfW0Bm0*ESqm=oaS2+7dhaMn}CqFyR^Q%fUj-FV0>Su8LJ5_o90bH'
'XinH;nWeF@k0#x-EX^6I3%%o-Bn8QfhV1A)yS#vz+m_We*%c1})>bj&(|ikwPcnLdxju>Mo>dg~h@1Ce)r8PCNedDKmdlHZWwZed1pzw2*8&by'
'0RcSEyS&&`#hygbrfu4aMG;AX3RMG807emCjg;b0&7s88ONiI`!6!eY4Xm7Lkn)hwyh&pP^mkgbe<JC8@@>j8U=k-}bWgK1p-oZgnKX@J!J4Qn'
'fMN%Q;%8PD**#0X0cI9eO0%E%=l--h-T}k(_P&#x{+ay_RA_wSFxc0ae?T(gMlLVni~`wz;Us^PrR7u~V79uvAn<dS=E+K(`^FR8@qFUtw5mA4'
'yjv0y^+8PH+=#XyO)Ij=3fce!r^hs58`@?~xCpL%vSr~F{>@mp46bHku0Ryk4Xqk?wwPc8GTfg7))-(?gw8;yAWvbeOU8N-_Q4Q*!4LLAPpcTM'
'3m{0{RJ~bV%`z4@EQ!{KC`l`LE-Ik%o<2qy%eT!gyuMP1*nOI?9FAVGO}!#nS~oX<^wz~1f%*cPLx>O-uElbO+(%?zI|4ez_t2voV7Mn21bjpS'
'u;>8<qIjsMu;a~)%wG|&6CY0s4gRDXf*?W1Ch}Ei4GE5PGdPiGS~K!1&06+NRl#;VWB$Ia8?t6(3#5eYfuRC}V<J-<HrF?}syATE#Hk&a7zVds'
'p)>bTF|aZ@U(?jLH{5=G&kh@y#risj*6z6I$299fxT>{~s<jB7U(0!qV{@+LocUF-yymm;LC!ind6s+Bx8ObQ`sX~YlBZPhPSQ}Rflb*?Blc;o'
'W#7_-ml6^Bl6!8>Mif*6fQGVtxubPj+v@pTt7qM=e*^s$c>Y6vL?`5;!$C(QuRdNSE1GYap^ygAhcwbs#j`X(I$R6_?#Q?%o+<vnrgcj*?uC9S'
'S=^Tk0Afv)CY*$y%8K`N!SF^44b_wVL8yl`X?DY_IgmS+Zg=_sN2OzM{=XwIzYO*c@fOTmUP+4O>E4oFF9#gb8y6*nam}a}yRLX~uE*N*&?l<f'
'5+jgL?yZ#g{G+1fE{)%`1D4Z?R(3^khp|cl@o|9<S@FQCsM)1eg0H`~1*lI<#Tyva3?TVqL;0T_i|=qh_yVHG|Eho%pVu_mGVk_}GG`5YY;1Cm'
'8m^AOWkzGRXL(~ByeUfD9&iSZo3V{J6u_{mBZC@W7fn;_MHnYxdQzn^bGd>^8=COHHP|2)wPp7V(Ls)HrPxEo?itMlc0NfF>wqpy<CRi?GiZ#v'
'z^4aV<q(2vRw>qSWNOA4vZ^AgC}DAm>C>3){{Mznt{v|r{f{x@TY(SLr2pNDzs4B<93_+iS^}v|0cjrc5Ef2je7G;tm=*UzD(lUoQYCs^kCL{M'
'a(T8Ll`}ssC&}oVWlwYH%d4}jsK}H=1~?>?9lgPw=OUXMahk_T<}f7|b}mV4o)dENKcuDL4sGI{GzYEHFPG$Fv0&##uFvu!GJ0oQv|!P?X!C?t'
'CrE**F{ZSAB0!4&1`_@}1zV0Tl2y4|7l|RoZ}%)A>w{5`z#788UI)BHpki!HucLcWZ!zV0ceQv%(>Y@OdMUGCd+a#L>||09AcpKo8JsSkIwOGn'
'@`XesVzh|IXm&Wv1HmVlFpct_2WpWhyGWEy@b)oir5REfJ1h4jGpizZp+Q*rk@%EhdIGk;eyJR=Rx*XnR=bZmZcIy#5F%2#&aiT@E{Y5gy{%ei'
'IKv1ebd#b}lp$Q4F`7qA)q$i`-BWl$XD=c^$ApQ@PiWwBX3c=u8RU^|4<{B7#%M*7H0I9F$I{pM!4XV5GvQq$F!YjqHj<|>SnIrSk>UuFL{nv1'
'c&TgEFMio487ZmxeS*RecfCzpW=!lG-G66YeiqJ<!dbx_fL6l3!K12Rm?#=^gQDw{2_q1Qsg+DClj^gfFJG*BZnwuIUmh!v3NNalJ!~N)2LsrW'
'Lr(V)w3lrj?;w}pPC$%#LpYuf;E(}71ZI)65SyZU5JB|#-sJ3!r+E2ZK71IRrnn!Owr$!EwtY9c_El1!2yK$@UFtN<ddFKSVKA(ybpI<&D$d(5'
'6xjjLkQ5aUt>$8b<SVia2KazT`ZyF*YBSs`8NCCwf#Bpz@&%cK=UA9LdPNq5Wf15YSq8igAy`}~1lK1p$SEM+(?zg+>GJ_qDM^#E5Vl7g5pWtO'
'o{`T1(e+YV1Ri(;GXZ*zE5F2*IUmTM$x^SocxefVSvBeo!tsavx8IGz+9CXJjOe0Jcw<MzPTdU?If+hG)I<dMC}Nlcchdf842mEM3DOF&=(9fo'
'7apYB%hEzao{{AMs$1i^t73&hj^#JQlFFV9Hz-kwLDZSH6hK$Ko`MRGjR3*wGN{cx6jf3WyM!m%YKYrX>bfTR?ehVpM!8pNtfEFER1~z}4KLXj'
'!(+-`7AI8(X5JrfYFPoJH2tbS6$!oW`L0SeXvC3y1#oUacyoJpz^6dZ32QXkwyFyZ4?xbu2zC;=%7SQ$I2~R=?^;}|VN;}8u%@*R<LjAWepD2U'
'_Jls7`MK73%;vz{9TeuILxdCoBu2B!)+fwxLWKig9DVDWQW~loa$wCZghTrphG)hOx*-SFDrJU(+#?`}Px9jnO-Kh;D7s$O(tGt|l{LbfjhlA8'
'=j{F9nYP5ByeX0aElb$e6kiMwQ5p_Ka?C3f9UsuQO?10(B8<N6(<K#1|FEVq`AQ6>#&mKdPHlI8IA*6N0>e`M0ufOKDk!(%xV(TfE+91o)p+1K'
'@PJ~^Etz>jQ!2g)UVz7cxS0TjCb#g|7lwK8qQ8{2YJ!$z`VeecGo9da6Mr(9&4LPsLFi{#_<>x#*-fXDVB(X>XZY)H_-h4lw|(@mTX~p)W8x2d'
's~Kvj1iL}cFNr+o*+f2-pJwyL&=Oh(0_}Zo7PkS~NkFUFDi)ITnGcch80yZw0nSOetI+1ClEG>M_>ppI4Nkh-+06=431`Qa=~NQU-)s>Q0i!cl'
'(aI5|&c=>QnGpR6ZJT1!6HJ9TlmG*~(ra^*+pQS!1QKJ<=|q)uCWuqU)wl}`*RdjkQx1HU7#tYZcVnrNleldrvojTj3+lvhEECiY2Nqu;d`Y4W'
'qj=v!)LD#!;)=)4*`3A)GS*e>>tk^$X2X(2M@+E#;a)}TX9Pl-st}bXSU|L_MxH(Q$$)pf;bJuh^)SrB8DPR8K6^ehq~$Y7L>)<wY*KoO@p@Z-'
'nj;COCzk3Q>XWJ%$f_8j<?JT8T|dV;%a&irrv~cvsOuD<L+;r0p8~4yh9BjZM`!#X#}Cdp@A1iR{Mq>~I(rg8yf_fDcNZ=(4-=luA%g0+C@QFT'
'2TnHhB$Rv`4hsZGFDNjWX+(D%*pBT4Cnfl;V)M}oo(d#8^CR;4NX<f_^@`2L%yR!keq;|s^=WFb(EM5j#x28<xJD*i1Nok@(nM#(^w<S_##?}7'
'gT*JXLhv71!#u27)N@1+bOy6VHZ;rdgHP6NW8#w+){++2$LnR3@H#o(5YwBATfAteCaK{FZ3t?^`?O(u5fX8#!y8hYtG+tV2Zu95Fpk9(X}Aur'
'eV(^V9i0;qaPyUg0de#iWAD(I!mDsomD(G#0nnWbELL5pfd&-*;MKu2S9MPHqYg4|yNxFA5xcDEIb!vMov6^rO=Y70+)cv`I!H*Kgpa~uk5(*<'
'Pa{COCo3TJ>j-$K_#Avp@4I7xw+|&t;qkc5=Wv5*fj+vG9tdWeLr-!JsP-tVsC5yNi2Hoeisven=8k5pK8<k5?*B2#826Vb4W>>kg1;OU2>Zw<'
'r9w35PYHw7P(LOLx@CUc`4>P>3<JjqbDF8#iW*!M7b}L@Dq=lhhF)R)@>B6y-tIWYXl)v`SzHD$j#+}EF=5B?jvl93F0Waf=DN6Aag^Wue#&uX'
'$5~M$vqwPtycPpT+i8Yt=ao2~qk219M|_$$<2;vMUxu3EX10`&_ujr0*ElN1N#lJmqyiltukxh1T?XstN-*5j76j}>2poRt>N+5KuH4zDRMMxf'
'GYFaLUs}LK0dZR~=&6*znt@^d01AF5A$0cbYd)5cHnHj+Vs?X`p7O*<GIlRZa$F?q&11wzOtPIm@8(gQkxvb%+Lm~|FS1r#z+baWj&ZlEb}5Er'
'6~@?_zTcs@r$?T&J@cLjY*N1lCJDPyd%xW{D(cN@@$%}nqp!Xj)UBaTU_`KZVPz<8J-Ksr)rHC{5}jyB&%(E^@@5VlsqL#Ricxt<WTsVBwOgG;'
'>%FIMIjz*ad?__};|A{o!G{4Q<ywg=;=D@M;fd}%$|>Fle{3zQeP=Fa)$oi(R11nLZh^&#=JYDP_G3MCE&$57;f3s0!C7nb6jOrp4M-l4O-8r%'
'k@ekny19DS129&yTgB?kvK^oo(#e(Vf$2*w{ZhJ&Q<Vf+p{i*&L-WOevC3be^=3d>xqN878BofLhw8x}QjV~DG`<^EoX~i-c52_92=VHKD5<!g'
'Odj?=S`;!bTZBqp%t4a~f?J`gIQPD2F$gv9I~ajH)M+qY2Ta<1DQ^P)h5}-4&AdPnYV{sy;t7_4h2^}R>So?bi;d4cp>bRIcq`lUyl>H+Qexfp'
'j*AKmL0RMp=DEBpCjWgF_tWcx?udzh7UEoO=(a#!g}jQacN&<oIXl)Nmwlh%nK#`Mx0d?kVCoIkaRNwjI`mND34>RS-%^NP=t0-DC>0Ce1EFN!'
'r_ab&ysPb&cM@Z^=y|yTLVu<yAN-b~mWKZVP)h>@6aWAK2mk;8VqFwY6dT+N007G$000yK003cWUt@1=aBp*EUu$7!XKZCIaCyyHZExeo5&mw#'
'|AD<ASQ1Ck_N8eMxh&w=GUCQ}Hf$wn>KX(^t}I3r$#R!EyRQFxpV=2_CDB*+TYX5xUGC1z&g(PFckkX^RebP!J@Kkq@=0CKy}T`!y!AR!R1cn<'
'agRS1LeikDv$XWiQz1{TFTAvA$|6f;QCHFa-u~WAR|&7EywsbvT+F<tturn}=(Rk}1%|~kmE6nfYFa#Wt=fpC^@4U3B@ex{%Dsvcb1bf;*x#G;'
'ClPsL2qAb&FIt{)hzCYefYhQXIcV}~S+sSv;Fa_yd|J2MD;7=NDyh@z$y?NUS8@S4-@SXczo)n+$+VMQ%ag<`#!4tb$pzMC?srjFt3fTy-X@i^'
'vY6P7*BGuA37)lFt|^E*n{&Ayb`ztS_0-e)Sr!Yv8gy+5E)j3ry4{{^`9C`@)^z_>G^9nc#Qt8+r=D4m&<XrWDxhEzJM^GbdVDJDG><>k6%U5{'
'dmjGsRHk06jvVFPq7lAX6-2zs>YV$5S4{0zuMp%k9#QakDY$yllnieW+mtrC(Z5Qb_gWm{8q@zmZw61{t%#r5-v)58!B3;#*pS3!r*$P^-NE>&'
';cUnp-#Z|2*-M}q%l63}hW<Vma<(l2hIPX$-;6_2c-_hP`@_IX1tb>@?9Ux4`ap1%N~?wr@7r~TFMrH<vuUvC$MN`jL{CH)3mlR;muXSPn2abo'
'{2<D?Qn1=7V%GAeOf$}cyW!Edhlls;t!-LB)d)5cT#@&uj97WkM2@Ju=&*Vw{k{ynXU*+}1$rLK)?Sjf*$e@dv>jAZ*D~~=jHZ*qIt2n|^r$>+'
'T2yH%RFt-SS@6#XaO5Sg(kkPgHXulsT~1NwhU79Wq3TG59+5<R<cQz+?%sL}g1V70@jqY`)wL+dwuyNYDI4p>#ni5k3Xzb@`+=j`sZzJ#l2Xyi'
'T*9||NZpC#kJ1l<=z&Y0sny9Mt%@o9frWSX!Itrjyb-*PSGmvrQL*Uvx~P01TVHKgl;(j4=8V*g-#G->Btw6eU6Hep$wHPKzp0dMG3n^U7{ReQ'
'9BciQz^gD4b=PKk5Cq2Ue66yR+uPANIlUR3U5sxd`6yWseQpcM^+_S{N@Q~uE$XKl7L&Y%UA58#(mu~4tTqgcdnnUHP1?eo2a0J_r3?O^4YapJ'
'$Ub(93ObXGqQ4i-Ihn}N3E5{BuHIc-CugJc%aifwOvRq2wPj4=L$>r^cZ`t19do28ZpYd`lNkzE-9FjbT{S_gNcf#t@6$=b4t132;kH*$bZM6*'
'5jWqdEGcKb9%}!L=9Ha$l?sVYuz7mCThQrEQ1*#H#mA?0cT5%Ngl5@&@LwcSajd4aSrHkcE|+){YiqB@Z<1Y6at>e;rm_1v-z-vcLWKd8pq9Fx'
'ik*m6=W<tW#IEtRx}C?)!aGJYNnW&2z_EHAE#?^d`j8?Xw;c}^qDktx8W~bLyf%rMkTVL6_+ta5AwpU|Sg8j&pva_4$w_^2Eg~~SW^SJR^%7+U'
'mz~Z=_6^e!dQhRJlC&UZ0}Q6LtNK$QCHWk1u3B1vqDP5N#Osxgq*2!(g>(OTK3wYj6)wr~Hg@`!ZxE5g35^GqE)sazRl<rirOSCRq@QKrGTBmq'
'-N?e{mlY>V7;qqViK|T?5z;5g=_N7>a6>}4Aw(jM$T*F!RB2#YB<ne^D2s}8%KdLXye{u9(U`D^p1!;~J-JM7uYSHc9VKTMH!+<``-=&vAY3b2'
'?>V`=y!tiy`P1da#|z}V<YuJz-^NVBeOP#Mk(^$AyuP{}CBI(X{IsG=e(aKS)@eD;LHdW2)1P_>X{KkR@#yB`#ixtg@x>_^pI-eky7?_;Q&e<r'
'9qTLahXP5VuF5A*)`0tUGfA`gF^uaq-2~;xlU-HuG9+-HcbS6l8Ppb%hO@Hnat{f#<B?MeH2#hBkFOFvJv$Cd!)OB5zR!+6d>0+!-w_KB1B5WS'
'Z>$!;(DX|=#t-g}40a$O8Bf_v%4T?Qu-ZL*rj0KKGnBPXAu%cPfazWWY35=5KDKKjg$`M2m14kpyvFU(wOVoq*p8IIvreWHv@dM1fYWu$BCA^}'
'9%+&2ip3M1mjNG!_>Vb;WP`!1CeY!Y8<QkF3(55K6wnlj4s1+I*}X#|`<&}6lBmc`lEC<12$yyv1>|x9x;Q930%^Kux*eZg{X8~jHq{0I?bt(a'
'm{2`Xox(BNmyFg=yD~l}-|2(hIr4N_R6JJ2b5SEV)K!uB!TawI9dJH_xZd`XKikd2-TGxjZR)b5%(G59Jr$}4CbLV`zf$iy1s>)(IXUWoE02+c'
'M8h-bvG=5O>kGkF5#2oI$^vPVPKpx1go}8$>tY8BK(6#4Ee6IP2JQ|PBB=zsR41r*^UZ0~WcUbcI#B4IuCGcf-HMX)#y@g6P_MS#-Y9L2bt|-^'
'Ev2KbQS@?hj<@Iw24f4J3Ut`{ph{ca@DGrNRfElx@hJB@y3omd%2VpgMIRLJGwZFs8X(jF@gQtTw+uNSdMmb`WeY&Uy{<}^Z~%&0=16Q?;vDsS'
'1p&+?Le2pLPO=Zet}2TKid3EuB%nEiGIkB3k6=8c$YTcI&3IW3pvH~zG~<u{(Fa$Z2*Wq|mU5@i_U#+{i6egQn0{vW$E(;@c~MWVCJk`v!%Fn`'
'6(Uc02Xz!kFNNw2p<QGj>D5zpb!4v!s3~h@F-_qz1`<YVQodz*unE97nwZ^Ni4YZoFxG~i6!_7ohdS{ET3>aoVSo@EMHE!8u2I<12mYy+=XF=*'
'BYnkUshDLVetvna8$n}{&8UfM(@NmgLMP`5nk#_B$&(}pMtuczYX``6h?f=<%~PI2S($m4HpNb3L0M)c!5wkDkIm5EY7ta8>#WdspdB)7Q3$%%'
'WBtnCXm?&w{xgmsXIL<#`e*ftI8p2c8(NlI{?8T&$rA5@Ft_pBeBClbvkrLq@jUr>bvC-B=50Ou%f;=*)u&fpj>orVI(T{MbrE&B=(E4plOYSW'
'(}(Qe=5`Yvjae03)3Ot@ZMT2Dwh>Vm$eN3#1d&}=Ih^6U4~K`Fa-d7CJ7Fqo33osWQGn<Fq)eiFZW^AywLI!x`fVi;dKU-my;1*e2?XeeCerpP'
'Ui+u%)#ajL$E)|RxT#+2*x%V<F4z1W%G&Wa1wonMWo+na%i{R~l_PZLT0Vs}$fH3L@wfFFMtMMa5Num#Fqq=-huFFUCChDd4;<(COZ>-gdn{?Z'
'TFDlFEj%R!Jy_OwW?w5_{hqz!mTK*>>z3PM?HpxGOL%5MOkD=SE%rA5b7`pSP}2&J@&Y;6K(}oxz0TWyB<SOAajv`7%qj4(y90bEn)<I}zHlgZ'
'YQI&`w-$Xl0<eO{ia@*AJ`VjkNSiEm+ye+5fcMS+4)BaZseW)mPWH<T0b{-J*vs2=Dm{C>MuMjW364aqbAgQ^*RSoRp?kM%*M;_a*=hf}#exF;'
'*#eWHP0H35L-T;9N|R*j62S}ty0=r;fqHr!*00_e22NRrN!_$Phc=VGvW!KQMv<nCR0U&zr33SA%e|)F-Suicr4*^Ib%#1xny(<Jlv|%cUQlL9'
'J*x~~<S*O##Nn6CP;&jl`cEi3TZvk0{Ek8RWW7UQJl!jFyPkFS`sUy`tC05U_ls!3aQadmwviEebW<!)eOc(4+v?qG&AwaFpg*A*`sccxfMyVo'
'MBWk45V$+Mw^<x=w#m8!d4__&Nc}|nAjsM)?w!@@yGiMGj=Oy%oY1=M#&<RKNwk(uGU&-UZ;Z2DISglF;IBC-ru`w&*JWC71kPQ6gXG(F!Jc@v'
'@^e~9-+p&H`nG4e!L-i(giT$KO@RdUO5(MP=LJEZ<9>eI!2mQ9WhF2P$xZ*Dkt89!oFu3pNkUiFNn-2ArtQ>qHUZ(ZeOTTd*^*)8AE_VeKX<+W'
'YWJDiBy5n@(0i6o4M)?J1(6kkKH*{Fh4nk4O6~i5{{c`-0|XQR000O8003fLzIox^vjPABKLr2)7XSbNVQF7uZ*6dIb7fy_Z)0I>Utw@?E^v8m'
'RNHRaKoEUL;vYuz!KNZ^(mqtG6sa^9l|mYnw5qDAvKD&`Q<hz4b`9b0J8R=$MVJRXmoszb=D}bvRm-KuGi&r7R9R}ulqYLoWt}Ip1iwPSXKIA;'
'1TwoRtwUg2dE}(u>-Qci13U67um)jgd5t`Rg{@2u-onNEkIA|GE)tlt&8w^qFR7AJGKvp*2Tv7NIO_K@Ro3P@x*LJ>memp=(}YI+g1^gFm{%Iv'
'0i#!B0!QK_%N?=&3AMps(C-y&H^SO@eD<0aK*FCrD7QD+LA_B%l5oFwuvvG2mCCO=n7N3d;M!Iw=T%9q?N-F`eDV@QQF>_A!T+d8wat`HLv7f?'
'zgv$BYuDfG;xXt$ntt!|NtU$|ax6eofv!|0mUa$5^?M)>6<}<rAz7kFvyG#>o9oFqy_!wtsW42ZH@~A1MA0yDo~evmV0c)O%vJ{KX@fV+mQTT9'
'Q#lU{3~eoFSuy4!6`>sI3Hz#yhD}UiAry$M`4f#N>DB$+{QmJKo!wuJXX)*DHv2NZnnr)&5+ctPMh&X&kLX@EB@P3KJC$`p{<?pdrq`2)P}&A_'
'Qt3ataM5rG?;tXVKxaXhxF<~>KeQa{Y{^FPIwff|b5(lG<Ab$i+>))F&^b;pam|c^!)vWL))5=4=Y6NWdqSNzVnIe}-8H^}pIgc-8*r=2=XPhB'
'K8I;)R9ghE+}gp8sa|tc2w7-yGS-(8ZYMcL?5;Z5qrQnFJmp~G4*ztavh|X~<)*dI@Q9J=);I|F7Hl2nZOkN7CPN)}V6u&;ywy(83$bJ=V?~bN'
')}Etq)fJl*pEIn#2JZ5Q>_#W(iI%rxYVJnPL3D?cj6Hdcm(k<b^9yJ<0c1r4D%;ZD&}I8Ftnmq%OjkLk&4;errbZ)f{{c`-0|XQR000O8003fL'
'?4EHRE(8Do;Ryf$82|tPVQF7uZ*6dIb7fy_Z)0I>UvqSCa%C=Xd8JliZ`(Ey{T?9y;o!lLG&sVBVP6s$S(d7+TgT39WouRx0wtYn))J|b)RLg+'
'f8UXkUBz~b4dVw{q>jgX@1CAK7z~zNtV%q4kZOYpgfy5+xq%v$=Gvgh@WQSF1BG?TGOSP-D5Qc~$-K!7mjXUE*q{yvgTd)Zp=1R~QZ%Mfm?Xff'
'S}NKvC=0XE`t;=Vq+9)<rRcU~QW$(RB~Q0ox|NCvwnDYrqd%71;7_}`;T3i{YG&5G1h+J`hn=2eCDR&a9lF;tT^gzI{PYB9P}y$~YXX=duZ;$l'
'DM{fP)!)R4LT#9$<wg@%AO*tm_bHVCIWsI}8ZBEp>RJ}s%sCcB1sB{TiLbFNMy|x5JJ=4p8rnq|yLx>WaO62%6j2{mo181Zo#=6_8XN&WQm+!Z'
'ans)B2TplihJ?rzm>E9+rXf#z6}Upv5FasX4Eo;nBD#trh_8N~Mlg8^vv~~B+sQIs!UvgZ@Och#^frdu#pLE{aSv~z`w^rve`=R){OoQz4ZL8>'
'!!P@-+>x^_%NE(EB~OWuuG3*Z$Z;`pO_z-s|M+tcu2D~WxEuioXzh67SmyNT%aei&R+jr2h?%UE8vUT()VeVVk<s;(?h>AlV8aDnNiC+)w$o3g'
'WcheTiWylIHKpk$sFYOhzK>bS2LhdQLTWRO`{~B}S-am|;`V2o+FeJ7sA)7c$rkg3nXx4?)J7Hd#`lKzXT$1jn8(9c=fj)x;qou9B|2PHE+Pp2'
'ueK($WweN4GK=S}nEtq$-bG9BFGp}0z<dVR^V!Sk<T@sm1@K}HcegLB8kbS*BsRv!tZZ`3Lp$@1L$t%+HpN;@m^W3eeIoc71cl^L!t&nl?FtWj'
'+i%mq2Hy3v2Y(q&qw5w3yj;w0S`Pkv6)hs*`S_Cf-qsxiVSy%FlScf*k{u}9e+TN_5AOrug*y;S@*}IPJ|ySQ2@3by0sEe*mG0yE4&pVoR$_fl'
'(d$LF)_EVZcAufvXQj+o35Mwnm^B5h*1##Cpm!%_*(U7O$~e`sXoM9{WEJUqTMKTHt4079l!uesx)0Iz7%w=ipBnSkrgXgQ1Fx9ckpB`8)zqZ='
'1D_$Yz9NjWt}%z-Ztu3|L*_nM?(M3OoP`P9dT#l2VLyfL%d=(%Y=<<hw!Z(LMPk3CTd6u-5FbY*vwOUgjn@f;c=v-N2G^>uCltFqIE-s`^UiAx'
'#(QrI_~cP%1T};N={fRq@AHu}x^sG)$KVx=mlXe%XlGLtM%zN$P_JF_0w5wjc`X&*dB@4;Z)FKvEAu;t+#c*KbA@BC*Y|H2MQ^m;`9-78{{Ucq'
'0lG;mZvPAke6S^gk@Q&P6vbAI3-1WQJCZu2mZbLfcx=D>o%^|$WoLnx`;PJ4;dZYF)$$KeO9KQH000080001DT``G%F~bZ102Lwt02BZK0AXog'
'V{dJ6Z*yf|ZEtjGZ*DGdd97Lha^toU{@*ix2b7tH(h1g4vLyS|C^OFaJjpb@WbWMj7~7+PC`jU&B3Tf!t*Cm2zEIz(uh4G+lA<Wuy>vqV5D5Z{'
'-No*=A0WqZu9R5H6CEt&ls!ma$BNCBNEU&gG7(MLeGo|<_{<lH4&o@~Ovz{}6}pG9FT!xmCTkW36OpFv<D=uF_c5EO_%W4=v$>3<5*mHwlk`_w'
'OhU>0cq%!=JTo(O8vKU$!BWg6OXWXSGV*2GW*1?2eB{Tmng$UhKp7=j6sx5O%>W%Uy^xR_q&kS^tmB<~Pn|bik6nEH$fhz4<`I;R9t5f7IL`6W'
'OvOt!9?w>KrQ~>wIg?mvSQ*7y7~@F)tX0Y4R?8>-2%lsU5}_BcHy{2Z9%<b1(NxZuOyMZmpdroS@e#xI1^auX)hZ=7XeFZ5S3#m*FX4s|&Pdj5'
'iI*&r4^pup`h-O=HpF;V!IT`fTw*E64w9!N^p9Aom7S)oWD*KrHk{jAhco9t|NfU#YTQs;qujEkmS(x-Hlijp?hNs<y7!F-5w2uzNW?~LDV`c0'
'Vjx~cQ~Q>^g73X%b5FUMvx(U2*X(rQ;o1)E6Fsm$<74P`^R`Z@sg0&TQwu#fJ6)C{89yQt(uVoZ5hwGdjC7i7rewMbHD~@xr3kDMySd}cYkT(X'
'3{BYx__r9GjioD?Ayg3qaB23IGZ&FRb@7?TE9FbK$wnjQMnWkQD{g5xUD?#FQ7*r2$BmIKCzYmD3#BW=;|O8AlDk&Yn#?7YjoFn61!ogjItyc='
'jbCmYBiFfW^1W6lfVTa?Fj%m!4|FvGOxg?fRgEDP73e}Mg+&lGCcI`u^Kb;_ra@mbQ6p}*nr)FJ=+~GAzO@A6NxhNvbW7BFmDh``xga|i_1qFQ'
'OWOaR>zgMlkU~xyd7R$M^(Yj}$yBh%A$#1d+%D!HRiNc~8B?r{fvpXKouNS+f?(ie3PH%<h%-42^w{2-NXgRzsQ1BuHGt6MjqrU+0wscwo+6+w'
'p;DsaIaITNpGb9L;!ZA;g-8R7>Ll)iu_Z;Uu8=;=fFVGO?uD;crWVL$pdk_}LQzbt7a}rRvQS2Iy~siAZB|aqltsZ=hzBYHiJ+oEVCmo69`Y#('
'13%C#p3PYL7-)YXb5RD{=>`)nC4zC|kV9e>h|n1_=gV(B@AB-*8w?2KCUH2;TTj2ayt=xgR>UuSQFOmDm$SXe9No48+h&G4J6njtm=e@DHy;h6'
'v+tr=d`Hsk^YuEBL-x>XdtQrLe+<R?gG_%Jnp#zA`4ufFT-5S2Jn>PlLk#RX)mV0Je(JTDd7YYhuBaU9A$yNH%GnhlBr5KxtaA3Ff>~<)HVzS)'
'oPCtQnQ5uvhm|Hf^2cxB@Zuhh;_N$(dh2tPFoY3xg%GWRdx?>_1p$!1WS_s{GZ8+DF=rRiJd`9uVVKdlm<zoYVj4f9Q-~UE!S-)L&4cLs{N_V`'
'^Al(9A`k`$J#<Irs7pB$lvS7VllW=HKBB_%L6`5=v3lDvKe&D4*^76$XMQ@K{WJIc6#x7D+@s%47k@P?^3KHn-fDa=*2H~28aRUbQ_eCE*mZiz'
'*4YfzpR;S!btj9nEZauR0AZdg<D7+!P>nNgHnzz~E$IR*r1YF;_;?~zd2BP!ZNputdn2^!TQnt<<V=e>;G)cF=-Tzz@d68*+|+{tmD<ylt(m5C'
'bFb{$N??Xns)5O`ya>#kOJs9AUtzsBM#C-NSc+n&EFNiwEze^jTUWl>Y{-c1cDo@nw$7g63Qlj74SwQm4J_phSnDGn;17&aV&q_<&Ge&w*%i-v'
'ZU#;!qO;rEBu*QfCuq6eHg9hy$P4fG7OO-b9e-mc*`iy=X_=W0>e$R#G7mo?+dOXZN&hzsbGI<&6otyf3rb2U1e`l6T%FwJV3t>qc24V2#)zCx'
'Z3&=7t9?_!!#wa(if{I(5p#ggPRTJ#T0a{f6$)qOz@`D!GADDpVRP`y1f9%hM(1ws>ba4fVtBxB>{wE|fzs13k4@q{J=%Lv69{knQ;bL@D*brl'
'l0^5eX{D2BI|hSTxAzjS$h0ZS_=y>KbJDrX*cT+U_X1|$&^*fRe19pKy+6XV=bdTB{aiAyhc!bg6YTrZPL^9+Wtm^FtKbPMn5{IM2nuUIltQ5v'
'n=M$nN@ycQo3Mu<1r0Er+O=%eI>x&zT$QbH$yfX1Y-EjI47|$HM*vDwnX%WJ%hKXE$w|9VO%@8_=P((jl0m0^G31(n+>*(wSv<>@N8rkZ+)-7o'
'wJz8h{2YUx)C)0MpkZglNoP-TtMiiVB#_n@)lkY!{KShylF@2ucI(+h&#-e)z2JeC<j|8Q**=lF0V%$-<{pGDzR!AmVqc*{&BkD-a2~I~{msSz'
'q+wowZ^vw1tO~{rj(NYU@W4XW%D3SrEAzZ=Bh7=P(X5%-;&<2E2XBMjeT&`w9_oL;;{m*9>7Z-3%N9MX%5`FMDBiEn6)>h%@Ue}i6(}lZUgk?{'
'K$%F?+HiSOjE5|Yk#uy+=zPLb<#eHf=pNu{4;0b>p`*dfwOU0!ourg9(+nNu-hviwQ{?whg<6#qX-uiWHf|7U6_0oOVD3&-FrCZ2rg*jAX?p-G'
'moU~hyAK^`b1Jlip94!1K=2o+vuch_z&eSIH`ZAJ!YT?-gK1S@c6hanAEeoffx5ET_g7e+lm%c=@ne*E<AmxCE3{*280!jNEh%YWYQuPi_IUJZ'
'xyCC{M6a^kDP|{8qZ$tl-M2ryzkc`2b)Bz<?)A6t7&^Rv_xAd`AKtsq?_dwSYR6t*{Po?%&lkVC+ne25RnpxsoWcU0*_3f!qPN(8aF?>E`?m8<'
'$8))5Ii2Qb$vGQ%{DdfS4}aYbo=m>hnKE;cuj0#Z-(4b;)NdGdd#gbmH_LR(ZPTFX@^14PIJTfZs6(e`lry9rI<<mVrvrWtYrD^@S61=zoRZyt'
'>UsMrqKQtt4wZaCf&Q7t-(V+^&5RjlV%jjfiY|%jS43Gv8t_ADZi!Kn<lY+#N~a$EPx=|VbX&%Z967W32KjthyjEr<6akqU<tLtP`e#lV5g~kn'
'(7GQJ=2$8>x>hS$tM}OpdQ*ZL#RqTfW^2%+Fg)#AGtSQodc3c(a&O<e7I8B*tB{+)t$VOXa&yGp?_E7{3&jK_g^FbY#q)AuLXR3K`p6aSGzkOU'
'aR2EVLQ(E}cYvI9h-h)sUr7Hx3{p*Jigeyi8yt6QP>rg)oxMrLv2Tyy!mv@iXnr;7of=ilBZ<rgR1#qg{kvT+pWhtt<|m&USH}rJ2!u_laVTdR'
'^P4J4h_)|h0Xb(LN@+>ASLfbm1w2x0%?VK!8K!uhi_XC1SzdK}+1gTK_OP~xhm(ct=*+cn*6+yqPM@z^ecHd%&h(rwPLL**<jbe$gF3K5=sfYl'
'**kJn?{>NbTvn(8&(S8>h<VlS^13?G?$wY6W05F<$0;o7bz4>vq1S+ZKhqSt@|@fWotb3pWgyb;V_Vwl_lqgECKG$z{Y>=oOtb;{$L3yj%_jn7'
'(k+t1&=N|Go;Up55bOe1C<saMdT;$c>?$$AZiczNLE$3HLG95}bx|s$>W}bO27!pU8mwmBGi<kN(VBSq4RzYvO-DLA!#^7wtOPuvU5gTfE;k1~'
'ry$D2jR<F6n-;vFWh#KVS;po?CcIAsd(h7?CGX2#v4+v01r#a|TbrX?AeJ*^owGg|_YSwv)Ws4rht9L5!s8jdT|J!(<Hue94y^~B|7smC&2!J&'
'w&&7DPl1!1BvrH^<=32bPS3rQMU%)TXsA@!f|R$#CVmWd2&MtvhN$lZJ%QFj7HN&BEFF=g*$l>@mnbJ-q+fOro!?`uvmb8GK~mXs{vadpP8tSN'
'IbJ2Shq3olP-CY`(Wf%h!WMx8IkcxM#Msz4IjhtL<O+Q~<zy9~(grBAc-?Io$y>f_*>X#+%DR)#;d2u6Jj{l_XtUW(QxH`SB*=L!D~=f5dEb1l'
'>GP26*>}YHN>f#!hvjIocYNsV3ZP@Qlyw!be17*oq`omyEuI}LJ<_WrDaSKbC(WKr_BU`Pk;8ODIk?T>%xRH(I8)B;d*@lsm2YnkFw7}qSn04a'
'JG?0$Mh`27d!=u)O6SIu8lAzJ%^jp@=GUIN4!rl-LTa}Gc|_!_pHlKroZSR&d8f1^sQ5qkPyOLLoK*OTzT0ctZ~O?G%@^|4?dBA`*|Oiz%j~O='
'k+agXlk?o}$}vM_if0cYzGmLuca*l{@zH+)P)h>@6aWAK2mk;8VqM$sn@Q^n005;W000yK003cWUt@1=aBp*EUvO`1X=8aVaCxO#TW{RB6@EA1'
'|G?^HXHa>tz`i-10BM?_c}O>#q-cR8P&g7bqgW$qNy+xq@_+Alc#){fc+=Fou}9?Lx$)dJQ53xwT9&thakdwgP`qR!E1II(vs#u#dSbbfN7ft!'
'tJ@tG*o!ypr99SB3#P>Ht<cTp^5XL1BOpqd;(S&pkv38>eUNRLu}U`Vz(Jystf+FKn79?{#AFR}O(83sQEe$KC0U8fi;}B7JqvM_OLb&5cIsxs'
'*vCVm*->T$-n6RHhBL=U%(QG(DuBPpM3st|@hZE#*zxq{j;rh{1r|`;;jpZUvIaxoiZgxWWf?PZUqi&A@v_U^H;4inykRM?*bcn4T4Zc@Vs*)@'
'3VMn`zRH9$A9<S<(vW4py?|b#D7w5bawSRLf@6^+tbq1Z10ZO|C_s|}-KrA(N2%S}=2X*#xaAksiQvsnead*%;8xt;U%3`<WhNlY>q3-S56Fuo'
'wYQe&^dOGB$M(kE=6kC|I&pcCmRxJ*Y$JOGi;$||<vXL6RZnGoeR;v~12JFaIaF2Pnw{>vp6G(0WufSv@1(lPOL=EZhEAFb#s5%l50uKPQ9Q+c'
'lId_(i92t6&~z;U2vc}vj3_55fxu$Zg19XH!|94O%*|gLgRSX8H(;r#3`^hS4!TZ7Q8(tgl<*95i>u^d2g`X;iKD1+GqSC+O)s#QfVjnWf=d$D'
'OgD;cS>${vvLMc*5|XgeMRV%m(o*osL8WbzSl#v8!Ykf^So=<lmGwZi5|H~=IG9^mq(VYDJ%G$}f4;5j(n!PADY-3l%S*igca&t7$zJjLP^1Hd'
'7Nxyp&oVzk%*28ZP-NAqN7d>gU1BA_Us`QWrAXlJc~Oe7mXfJ56x>|gB_*Os*@Ie{Jhobh^pF?@Rn0ZqNm&BuHG~!;2$*FC|7ZyamjsxfNBdgf'
'mNV!9w#&`-=I8$QL7C$ac1u{;Cp>;3B4n7WOys;Rn<VEH5l>r!SUdJz>=XaPIr<M<ENR4jvvOhYdgjA1`B~-}%8Jxrk129|`LbqL&q-U(b-{e4'
'Q$=n7^|5I44gR&l{;FuiakCfADk5xA45MGK8R9Xc9r7bX#fu{-9<w}p@L6G}3Xwmuhk?Rlw1MvnwF*V?Du&VM)gSTQw6Nb#bdNDzRBSKgi<9LR'
'zMnyiPxg_1468GqgZ-SHm34!iqF${BXbKdDVsHQ<3Ggt8H3J#*_xTJ6(y3+wX3>7Jf4~O?J0rpMK+tK`SS(J*gW(yBkoAb$`W|W{B^WY}0q{>~'
'`27scf@J5AI6g~Jt?V*;zGc6}!9r|E07+~pUReP`Lol^L5{A>KESx;Rc>Tnl-4Q`Tc}wgghl{V9kVvop;pCs-3A}A&G{1=+$WcQfqCsgj07CsK'
'3YRL$#8E<jD*C)z*`-d)yg?-*I$qb^o!B84h%N<7H(lB)EmfCseb`A9D_x2N+aU$vf_d4s#rpFuTE`(BgvqQ;8&AVMcDwe}X~dPTR>%ZhE;!X4'
'oeh^=TkYh1hlndww}WS9bG(^MTbCm|Ze(eSLtU`dag`pB3OWeX71b@YQtVAk>wNvyHN{bs$e(i>P3vO0Cm`=oA9jAZgNkeOOLqwVs={xrfqLHp'
'bw_m?V(uKR$u_;xWwr0jN>`qs3*GIY3n&(>sis3oK*c2|Ml<@opkbM%Tn`$ykDaIsO}7I~$g|oCeURAMlX1-Pn9_#S?=kfYoiQgmk06QA*dZM9'
'8=)37#>*QjA`XIQ9*tXLwB}EeT%;+N^{{Z1l`JJ#3z0g<C5Nt%#bbubnCdU`Y-82eqadloMhm*}m4mY$NCBAd^%mQ2*1?IuV+i{wNCd^v#P|F9'
'=qbma)muLj0A>~0{}ly>V;iG&#AQ>8K*$755MuEhh24&xLJ<1l0)mtZ)1cB?M&hCZ)l|nNQKIaG8w!zx3MOlK2;5+5g@I*1F0mMtx6?I$_sTXI'
'$@Gp^qw8F-1Xo|;9KUAcVu5w+A7~_D0rbZ~K2clD;Ehe4X@7>E=e-q)PG-V}Y9L3O;#+4p7y3%qO8DM~GT&b4z6E;ao6b_~IZio=`xf;McR=;L'
'ZJre@(^ZEWUC^7#BB_x~i?l`K9J8-qbJ3urLJKDvy*0(d*RQ5{`T7-g_XZ8L*#3kZ@DnX7f%aTwU`y8enX3xXA@zNHYC2WJgF^JB&EvceGU|l}'
'?Oz=G<VqWg6IdVBtnsLJ9M|b~&z9ryqyi^A^MY|b@5Bi3d_iG@mpj(SfZ@z65RJpm`uiizleNCr(Ag$}0OpZtGjxi^#iv0k+#cG)MBI~c!gn=#'
'E2O&wa5^#-@JRnmz+UFPC#{|%vc%dV=V1f8$om)|Xl@tygpE-ruv(cIK#vcPYg@V=xVAW`WwG)oftn!|%hF89E;BrrXwx^_#-~impk_9}+(QIL'
'GCmcAO>ZU})Q6~&SCN^j*|0DmvVno!aorEB$`h(;ntEiL$k+AdNb|$-rM$MlWq4*$XW^Nkg?Ad7^xosrATan|)W1p70hvp3vq*BSKy3O&aW<uO'
'H|oR1o6^RJJ!+-Avcp(NHsaUOj@Z?t<R_F&>@R5R0|2`GrE!Td+sP%L0bEeaInKUAK`#i9(kS|B^i(u8%L;8S-Ot%P=D`NWR(^)fKahBQuV-~K'
'wW%i+1G_+=jb`-*>F3Fs<I%aZp|%)fCc~>m-exc?ItV2VKSMq+jMzFY9tx#6EWJEC)5+LG^K*J<`CX3q;h=3f=#L==c7If))%Yf(x~*WXMfF6='
'7!CgDXW0y=tez2bA>8`n#m2CV*GA)^h3EG{ob}*~0g*tLLg5I7g<Eze12=H-!=_{AV}+^p5?Pg}p%OPFh{k>oJ!?b7&Z=i+z8jI+z8H=*XY>r2'
'v(cuf$v(XPw@EgcysKcP8R899wiA-$2ppI<HQ>umXm?mt7CS{vPI%R3Gig;xAz5ll>!P{^UvH9^zrKC<>xWm#+kbv|^D=q&;^W`#z|PO>oL-06'
'J9S1!cfzCRel&v~n?krV+|n3+$)qM2AEvJ5V%_yP)&&#qcyeMC62Jf0(5+nu4^PqYV;Tbw{i2mLB}!u3xG8)%`nn!HCONuUJv-oMSl2_R+B?ex'
'bkq)y9+SbTribi62;k~)Y7Ri_f=dYaXllkV4aM_!4xw?Mt!?H7+`rJDT+tXTbEm8)xhm$_w2Yyo-Y8)zA8uc6)~Yx{Mpkh}=x;@z-OpxYn?P-b'
'jj8hey+toK?$8%YO;^y@)XFwGcsFpx5&R|fo<vb9^McruyE}-AacUE@A_MF6YEhwD(~Mp@DsiBgD65d~aSE{f_W30Um`}6((qKVyy|r)XpgdOv'
'lky1pt<j`7)@TR{?OrHqVwOT4r_5#ETlT!oh~&{rYZ%PbCvI9{2t0%)^Y|=?{v)V<`7sEQZW?m#4eKa2_cXR68y#R~)2wa(DS#SUv+eLP*biHs'
'ZTJwt^ws#b{}f_avc~P=YH(^&|7<7Ep#-7TY)4P=B)o6U?~u3rev5zcnW(%SJq1wVTgL5>3c}?O=O{94CxeX{+%V|a1~ZQtm>Yg<%^T>|crj+b'
'gmdB`m2ZR3@q!y|`%lA2*bn8baU}M87j}el->m%G=`&V48N^0WL=!unQbpgV4$U;r-FHFKvdzqh(R~5qii)+;f7>INEje!q+#(OYaL@;7<{5N;'
'vVB;1-vdXDLOg@hJe(U2;|R@426VmqOp#gV@o=FN0djC2574hn@eEP6ccUKZk~>Y$_s--`O`M7(GVU|T6nfJXopq@m7d-)lT&m|Pzo!*)qG2s`'
'smIOdD!>`puWwYcx<YhD$mFiOwX;u<XqYQ48Kt6~s6<}auWIZB>(fz)(tMML-tNF$1bj!{ATObMv^ru%CisTE8mMVA`?N2OsUo(CY=m*hiBk1n'
'D`H1}BTRBWvbL&3iaHs0F&6O8EW)#Wn{IN5bH!^Ughm`6qtUz6!A`x$u6v*SFHlPZ1QY-O00;m80AgLIYV7+K3IG71CIA2x0001CX<uV+ZE$aM'
'WnXc1VQgu1c`k5y&05`$+qe;b53v6OPcJ83ZP$B2iq;neZF9LIKwG3~-i(F7l_;BtEcr-kcU{;2duRA1>chL6JLgb8tSxdlocVEPI5XNPihg8@'
'i5;)GR6Hj?x3uEwK=LiiKTA?Jf~YMcg4G2ROw4Q6DnTp43a-drG#f#8izte&UX-HQkt{1))e4qngzxr^P=wZXqp0FdEw5f&y|ANl+bUjlkL@k+'
'%9)gw&z`Dn#cv&x5AbMb)M3x-jhpyOeVCEAw5sTB#gM;QeJE&++E}W;Qpx_&6s($&pE#=uPe^%|H$Wp<#)YGP2daIjXg#KpfXa^gz2Cnz;~&}2'
'Et3klUcJaGDkb@sf$_bREsOp3$?EC_fsbT&Ou+PtNF_*0qI)W8&?++1IaO@a2s3SmV0rMk1ElCC9AS67V0F$i!AiLz6`bZ8CU1bdRyi(MN!v<g'
'CCwG^PLU!BMg9mZ*!@R_x=o0zRnI_f2}kJ#m)qs_?<t=Pz4*`9R)gmAirh9$<<>R=WG$<Ps-ZmT^<p_A8<sBdyJGm9m}MNnbGn3pKB<&OOdIfu'
'6g<~R%+OZOuS7DtQZP909U6R*!7N(t7Xk`Xgx7M<a+RT@WP32D*dCO7UU@^CxqY=6twx4r=fCrGvZ?MsxQZT_0OM?HSj20@b_p~InZN0sraeON'
'(zgf-QGCZLUNcy8M+FC0k>r+vtLu^GGx`_xE^oa18Ar0dvB+0^FX7p1x{ysPz%=!QWmFOf3x?E+*YRJ_{T?=tdrsthCXJX7jfiI6P149Igo9wT'
'X;e2^xF?KTk4d9X5Y0^R0?B~9Ruqm3q#<U7k!cYn)*9FY9wVS_gB^mrSeiMEjvP>WnCVsUo3OgwF#(drnoklcR}y+F_4vXv%cP<ZN#>(n67mMO'
'`9rY$lK8moH*_ChQ%3*pARtGr*@+YlllriXtl+|UA?(rPn^w9zfaI-^4H(O;ABxg-8QG(dYy)1UGRru+aS&T@G%%JH)FZFc-jLq@E8&5TQ1QYU'
'%{@ri+&_uTmP)wZh+XXL$1Ls)=!_Bd%mAeKet|YJqMrxeRJ20hrxP}EDLmX3*^bK{RrxmbAPif?QFP{^>9lQ>yoL-Ra3^mZN#Y_2x{i)t+G)IQ'
'cU*$dCbG%C@941*F*K0=-~~;@2r@Dd6LKf)<L_FRqmc)$Q>HRrj7-EG=ah|j6(7j&?5}%Xs4bi_+=<I;;t*RLfFL$Mw~TML3a9Yibfmuo0+)h3'
'0)kP#X_W*A&q`2)JqH8!pkQA~>_TlKbt85l#(!8rG@qeq*r*DrCfrxOu-e&5PH`yHG{Zov2Nxz{y(V*T6OEr4M%#Ha;ZOi85?;oDjC3BmP*@M4'
'HHRLEZ#ALS+(nv-vp#;#keuj!Q$hqkvx7z!(KBkh5jj+_^u!B{1OQkF-<&w~`c!WMG$(<(ICyS<kDnG`3O%p<u_yCt@v=P00h%GuXUq_w-fyv-'
'm^MT^J+whPJhnnN@tfG86V8Ka2Oc%TzgdOzmi<y2v>Be$RY4R0Bn02b$yZX0?e^-B+2JSE!N?3rAA~p#r6bYu1*h+CV6cx`a$N<$t-YfUv942U'
'o8N(ZTqpkS?#?n?M&X=s#pG;4zDXx{`_gNDDu4&<JWSn~DIKQn_|jbRIz>jHj_1Wckx*51nq!B~$z;<%cm>IEO4U<mI{n~p3C=lQD90%N+nEI&'
'-T=YGSNFar4&X5#{M4!cg)TT@j{BzK`C+kPDvo5k-&YJ%aVid(x#MM&kQC~9tU<Y8k+#{XLKCT$moQD8A9SGDjiKV!c>~y_G%I?An^A)}VuGqb'
'fr|yJ)q~u&3jJEMu^Hw)Sg_Njv$XU0m?D6-1#c`YQ0+)2?kp}C*Se*eF3VeX5sx*yI#djABV5+jf@wi&aGx{CI6<XLzcmL9is}`<HvE}s5WW}t'
'(2U)21ID{ptFv&DYe<Z(8-K-F^iiXA5_ElB$i(Ijq^1xp<B<Dz;A-E$!XO%-*~VcsxshJKe3kgdjcP^RZ)9~x(tJ*PikW7Ko7K{$0;Y;s|DKVT'
'$!vI_srd@VO`9n__M=4<bRW!q(JHt!W&`T5>gkc6DU3!8dfDYPgMPL=WZ;s5>9sz=87>>%C8L#LgdQ&Tj&o*h=UAhqF5bbKaR46@+rApamI89X'
'TIyiDB5OOofPlJhRi=xr85xY40_!H=c?y0{v<lO;v103;Q+A7?XJG2zc#m>hO|%kCN6wlwcF0${F+Lk)EDi_TaeZcY5w4P3w(`C7^%_~<%m)3_'
'n_(}qt+Rre?ld-h!QVfLmN}8ey-xC9^3SHmxPV_?HdHJt8O3)8<EHI!&66EY-@SzLl2_Q1QCm1r+quW<LPx`TgI&;=-ZdQRQ%dqfUa{nh8=guW'
'g3!MMttrfEKw~^6-$~5bUXgeD3$Jwu!7#Q8&FQfKh93xGKfx`>1~zV@7e?8stR-DPKVM<_w&r<I^lLS;k^+q3r2~f(gYe0E^=i4iNkRv)>iT>t'
'HFg+lV9XWk{Wrm!Y}vK^%Wm0gdvjdR#-^<U|DV23F3|cKDeQY+atrwPQnHWEsJt9U?`)>D<Z$ugGKR0sz}hGB-u{YSJ`y0Lqfw-bgQ+-$um#j6'
'N==fRsjmW*kEYWlM2D6qBotDDV<-5xTA6Q?(?d9DM269@ZkR#=mQ8AV>Y=KelL1y#y4>#a37gGZGdJpUG!KRmw4Z;0GckOP8Z;kFD9p~j#X6qG'
'Su9!~DvMFFUk(idED{cgeuZUN=J+ZFk_#22^u0~b<@D}76X<ACYoP-lxrdBntCQ6l!?+h+5!eJ|s}2Mtg_5Wk^x7dj4ixjnh35hQj+<4!rI5MX'
'<Jp;zUfxTw&hvOQ6Vr@>tSMxJDY6NrT(RBoJ%GtHV4~kqe<TAt^&k5-hY<Oy3^pHOID-sMAB3Ll9rchg2v*`d(^*%!fs7-L3zS}t`a}3JJAq8R'
'R`Fmr?beq6(p)IXm;=69F-%_johLTZq!;yrGCFGEP8k??cbpTEQdrZ+aA#ya5xe$>LHTAikR~poE~i25-(_aHW;OqCvCvKxM`gL&vrR7m8rnh^'
'eD0qy7QkeXfUKt(A?oe{3}b$We;q{8oWozt7<4v-10@dEZB^;%=w_JDr~=|-8c6ct{fBpBBG};xB5eC2tC!1RZluk8luh+Jm==_UNz8Pb+h<ZR'
'1(^|m<|vI&3-cI9tt#h##CM_yg+v>H<I7o;tmoIu<#Kg%nJ&}#Zr1x}X4}BljJ&^5sXwgrU&7e0Df9pDs{Eh~H9UO@+GneYO-8qxNb-i1Hk(^b'
'Wq2c6pUs#8ES}DE#vKDNkG#JHzP5*YTThJW2Qe2`_OFYJ7RW&j{6A1j0|XQR000O8003fL8{sM$WexxUd^Z397ytkOVQF7uZ*6dIb7fz0WoBt^'
'Wo>0{bS`jtwHw=x+cx$+K>mTNeTdZ6s7|xNwr(9DO{Qpo2^vhh3oKkipe1^&jx4#NG#3o!-}f9|B$A?RPny+3MkcAlbH9-Zg5bxxt8$i;yrG-+'
'tYJ3=yA^+UQCFmHvMu9nLtFf{sT=b41Ig;$zUBoECC{HdfA-s!5q4Mbwy3Vij^zdYI-`5>MN@Yz<D_9bT2!Q~+PVXPc|v~M7M$$ryek=jw*nwG'
'Ez6%jtE=*!tnWz+Kr40&`<==C7`d&R>rGkT62O!-q-Z(W6b)}l&fxd1s1St=HUokhz$^#?L~qm7JCdfGuI(C@rfM%jtEz4VT^v{GQGQ=#Mcx1Y'
'gV$C6qpYv4fX4oJ&HGor?b@Ooes=49Q)i5Gwf~;B+p<{e%|F3UHPhbjk#B0^ZFL`$ceE_&x@0}}v`7^bQV#svQN28T{*1u$_QN|leaZgpKwhyL'
'=A5<m(2ta}U)j};ReI#zAk9xDt<>lx<8^t%@(X&;8W@Yln6L_%!5mB!9_LL{Hwx56oze1r(Xg!T$I@<J)-+E+J@%c(WCLWcX?C6N2Rwj_YnBS7'
'QX^8lsiB1Zgg`l^NHzE4Tr?VA94ShmA}?C8;dt@BE{p75f1ttR2X#}>Qq7OQ`?I5E(e@t}72gBx!%lHOBxyl0qiTUpFSL#&SFF7dV_{%`ZrBER'
'j?xPv{OtL&oNY*&EAGO&YExW+p66jB<vt%NM?}uvkab;`bIms}j7p3sA>vbWM1fT|Ves}t`tFyHpMLrEJpK8n^y4q@&o6?Q1VNO5XDIezWPH^S'
'wQo|EyR>UcYt9e!;^Jfa@5@6ljqffl-hO<W{(64-pYzM9uT0h`VWeO-*|nk%jd3n)P^k{PLvgS?%{%Gp<^&uKZF|j{U?nOUax*Wob_poNXum7P'
'S8zX<UFA3@_!zw`ik%3QI9W6Bt4wc#zJL<#fD_@v5L&@GOwUOPaB!;?=*n`1Wa4bw+|LJ&33GvY!+1iZInvZ-Y~PY|@dv#$<v3;lP%7~-j0l42'
'>tIfPq9tdsGb6Es+2tmXyKX30S3V~XfNKcQqvib7Y_@s~R>rrF8h_1qKw2<96iB1c!$8jk5vsy0nJF@g&G6X|WJmA9S@JR-%aVKrAAR>tWY8zk'
'P3V3Pk_b2b7AfC?#m*mkHAr4<9>0DVs|Sut3sjhd4)-PMw?*D=0q?&pdb5xoFBLPEjvJqCS#h;(z2C_xrRI|q+0p^}3UC>yPkjC`?DIk(@gjP>'
'GY91kF#3Jqg3)}KKlrEtFkWOI4LBytApBZHQ=ZZ;FX~{Lu3!hh!9w&x4Q;d8?b#J5j|{`f?Y79aVIYTtL0{y60X}D0joHt_h}cQfRSpN2@<VVn'
'XE*3@7=|}|IV<2_)No;5IIzLtsw+z|ZD~6`yv>0*3uCuWA3mLr7r_ok7PTN#1O-;*%m}}-aP8FasAxHo@I+cVzRztS;I_bmfiZfeL@JR#!=P(7'
'XWs&if)#ubGz>yp#)4>h_Hs6x%~v+Kk;Mrxsr(uYPV8ow)Yt<_?}JJsVHu<i?q@vJdyhFLqp-Igh{>zSTqtdI?lE6;l`_q_&s?o_y*AHfWc6#h'
'^$l_)vq9e!Ijd7Sq1Qed(XFbR9mF?GRv@tmkB2Lq$EEX$yie>scLJK4gZw+!5y)P!(t*$cIz+;ECEe|HU;~4L)Ss0C6Vlcq*{4|xp_=fzYclq='
'^dJO6J}c`z8{!~uDdbcO0ufcGIEffbIjB^X^H_y~SuBZJ4l1xBXBa<TM>-Hzxq>uW8TXDC7HQDEn>JVWD$8YBtvO_5+B#O_Dy*T3mhB)16yzss'
'idnt>)-qCDmq>=Bi0cmvMZSWe=j7m<1rp?#G2jg;MaMLf2BK-tO5m4xj|DGqC&{OHyh{{~;S6kqH_3i`i+^~a!O|!)u>yKcfNyLx9kMjP`~eRi'
'gZ-Xa0mC3`VQm&4`~db_7?@5hZWx8qaUn>S?^94S$o}Ly@#<L?*+Xs+^(c=ptc6`sX>(C_5led>%adaj0va5S)pH2AMT>dBK~0xMKjyPiIn4A@'
'nilXHyj*X%1}5Dubd~?iqw=OP^*F{x5g>5&EVAuvz!WSDRn^h9H3;z9XF-GJJJ@<G2dtpqCz@1yAC+%>I7tai>%Raayex!|{e9T&bK0`SLogr-'
'p~l$yaja$L&w)p0i(?pTR*e-1by6UXE=&)~#VThsDgo32gHgxzVOV_F%Q^~j0}(V=Z3Cs!X6NLf_ZDBJN}+x3TlwwT;Ip-}b_hPfam&q2w{3re'
'LC+o$u}z0Vm%}O;#s*%IKVg&`58CcPihELBk3M{OW?}DDX;TiLqu2-;blTVlQ+1)|Jsut{H+0pOCJ<|)WGc&5L-$Z)@I$Jh*Zg0ULXW6TETxdS'
'&r>BIje(vfaDzDW!!rlB10vU)(n3EfVheIFWMlV7O^zjrUZ8qFt&x4c#bk@`VB`d{<to5f^bm&K7ErWoHh~kD6OZx?KzO&`m!iW*oBI@!Wk`yr'
'L5&o44DhBdGFIPUb8!MlzhVNT&E4duFqg6$<=Kb8FKRd0RP9R@-O|$7et}{|_4X#p6;bY`RkvHS1}ej<@7B3HDWcH0*c}>?S#=I*BRE|}DzH}d'
')-?^rL_`>t3HC@G&b!?{#MecD#|d_!D9&AUmlD2HZ)UBBFYBfNCN9iA(<bQqEn@cZ_v-h`=(Pm@tvf<RXAw!jzG5|;(}>EFpr)(i6e^E^^zjKX'
'A$<bNV9gDJD%lOKTFES6DOq>;qrv_&wj=;_Oty7>9qaH8e@mDYUc-bbRF{aRfjEsIZ9^+5VyNNN0TR4X@Fk*J#!iF~1t+&#26T0tZMxFN^it{`'
'75)8^wTN=R2a(-(%$X_KgX;hj$7XQ0wSl~^WuKwC5%`M?L-bp~-|Y4r-6;q5iLzA|>(=K#BXR{7(@HC3LXM>DShl5Ak-^TI-+1IUw?qe`MNLUi'
'b9kr}Gluex;}I6+Sy8C|4saI;p)K}c!WZ$2n7o+3h$7u{>%-gM37*nn7NjkXY$*peyq?WwXMj=z^*3mN@Q*S`lpxJEA%h<#dCj;6hYSML<#NOW'
's`#d>GWdZlSnwQ0izj)yx!nm3M3(5oip263T9tinU#K@WQt<C`jwr7lBiFZlu^z}_xAOTN5+T`!?%3falYXo1K%<u~&clhosz1=NJuSG6vG?sm'
'nzRY+cC6haOG4H;Ug!dh62W~lK*ow%B`I;VN|e0B8Qu+rAPceMM4A0juP68}P1HKlIV7e-!Y}c_QwzvC1wCEJcu)=TqHZmWZfS${lX-@tCF_GV'
'H_M`l{fwad`AN)>zAAZO1`EFUiiIy@_xvFP^}tg0rP-wKPc~uu(YdOV!Z(&k^<$j)>PHH~8I*eEd@IOv3m@aeSU-xqh5G9+br6$O`edMHbaopW'
'r?@>s70dHf;4FExC|MOs8X`MMM$c*NlGN8uQru8*X4Xl%rwrOasFST!5yJ)d<(Zr5!%e9^sF%k%34HJ#Or(a)CB;Y-Mvu-n22BdCq+WE7#EV6='
'M&b>5IVu61ua{%w#GDx#{H-({F#NOl1Q2?6`aEC)O22kPbLQcc`2b}Sw;rynruc2qZo`0sAXpF`hQT0?yHKXTm;m;8v}=eDs)N`GP280dJ}a|+'
'mtI<&j?r6M^p|6jwocn^!>C)HPhQQ?kMz?gnCi)1!q%P>yhZPu{eaxn%4a*ngA?%j5XFbyi=4(i`%;LrBgT<X3uGCT7>iRpbPQI}6mm&luAju*'
'(M_8Xz=e-)gWwILc*lw_G@|kl%b5)(noI+A`p$-AV6YNV&0y(NT{VW4NSEMn+P;0{aRX1_Ha>U&49UqdP@GMR+`rJTWRaI0D)8%`h{nb}Ayv!V'
'6iZ$dGEGsD`PB(`T5x_k*^(@s_qWuRNs|1PZ0w69&Hj|@uO)graK{_LqXQcs%+HQzeln+*%O>!g#P7h4$EFkteZ_!DKNtB<Ul}CBU9+@+(okNN'
'G|!&YSJ=oL-yM=El5)#ED|IUphK9>aDknO0i!J(`4Qwk5K1d=&7eiA<_;msHl|!AH(GKEpRTE+DeU(b^{0#e<P|y!}B}QJb8Z5GBm`CD9u5Px9'
'@S2url$jIWOiE8`Qbo5{zS;{E2gt&74#JM`dY)0=*@?cCrjy#s(B<`_kmy{2>vf#<II3KM&*uy5BF~cV{uSe8teHF#{)T*I*PElT(lb;nTTJxc'
'oV9J;(o(MFcBmQU;SS4=C*r|7ac@{{r}|9FXbC2L07+bPP^pvX@dVb^;|pr;QI|mQDaTz}xcdCTR50Jsmtu#0)fZM5E<cvDl?>sIkjDj$?wQy)'
'4S-Kef$*_!Y<hyI2+m_fkB8|I<2`HqxsfG5`>v(@NK|5~EX02+$W^otcpCLaE9+$R@G(X1nYP=?=Htuw@bM+j_{d=F!)@9)BUAg2T|UJ}9LkTv'
'5MJ`l+^ZmB>r*ObfxHm~dJnJ4=Ax1$kOz|zf3+-O8FyP>i@U9!B8-UJ;jqY5A%XKaT1+2$eg@v(`>XtG#`d}syT&M@8|BJZSvPjS`RAA_r>q>^'
'jquUM?pmbFwG0E46B7Lw+lDQCR$y7-@e<S3Z_CDyhdq7r1}ceN@MCfBTM_qO$+-9O3&uT(2k-;%;G*edIe+_rFq)#*PCVjKt`Vz!p6t^`|0(0R'
'91nOhU-2HDq<S{`Pn5OM>&CEtOxE8CS-<}BvVLNYGtLJksT1w@!%~5I@N=6v2L$Tq-C+w=W8U_Sr%~fp%HfzZ4n~vMjLcz@$?-vLG_R)p4^T@3'
'1QY-O00;m80AgKrRY@6H4FCXTF#rG*0001CX<uV+ZE$aMWnXh+Xk~3-E^v9RT5WIJI1>I|p#OpA{n7w#)=85h+g=XaE80bYz1?D4EDl%{3N2AK'
'FOsN|loKEB$KTA5lxV%!PTB>!u`O~m9L_xR%#gF$?DMu38B0}VHBYT!xzfz81uIm_3-;y%+sI6)=cy`dWu%p=Vs}d4=7qXDefs3-lg}`=RGBE4'
'G>jMUGl;W_)mp&>fwJKLYWW`l(&ud1SQ4*Al?g30Q&q8LJWWMy&C@3vUdW7F5FsmLd6f!t%2@Y~N^D?`C?#mhEB5|<S&J2DUzWVeSRpF`O3h}o'
'r%!UNN|q#fV;e1!gh|j}TbNWS%W)Y7wWm>bTgz${M!c!EbM~vWLi3^>Qg1U}S(%1G|Kvve3pJXv_fizu-07Ei{guX<PoJa(H-^12M%YiP(W!W='
's$8y~wbM^tJ$=I9KeBdG$ff3b`-&}<D%dsqtQ+B;*Wz9jX6%8y=Sw~|rKwB}uV1&1-t)rvXSo)_sHV!s9^T1YS&B^Z-G_|*!0KjMfDuYzIjoYi'
'Hy=N|W)*EVx`}4DV(WyI{60xxPiQwyQVX4YZ3M1ORu)B~)t+S{=S^X+ujXteuFvMI5ZC96lh}BFDwzreyZ(DkyEe2PTK-kovA<?2&u4QsTPt;&'
'pmVifK$}E0RxA7^Z;E2}1r(w2PLLJ%_7yW$v;VQ*R3!##;*WD$S|O^HU0<JHEKpmVyg@x5Hv2_1xoAq54N=91);T*}kbM^T7cF$UYobs2{R52y'
'y9X0C0zX6CS1zzkYcIZ9X<-bGARD#gk5PAVdU-jb?}(f47z_Wzt9qA7h$G31j|zNlJ6sxs`7ffvb;<t%V2Ci@fte}{c&0A>Q;K5F%o?M&G2p68'
'Ek#ZchQdA*Otmw)77Sh`mmuB>M5qkJg9QMM-rY-M5iqQ(pBDf~1-rSa!Cu1N+{~T0o10DW+nXDP*mQGa@_YE4qpb|!U&&>~Oj9GCi0n0?Voghh'
'hej1J)yV>MTLQ2dh(^0cd@kXmfWM%k)Bvl9$pDtxn$y-YVO_W)QP~7AF7gBSjE3^YQb1T<+`^)gwcr_kuT_?EWAP(Itk4@N?r2=&tUQ}XLQD#P'
'QYD!vNfy9C)}%ByQz^Jg1=^8gZ3?x*za^-}S7Jv<*%+CU_f@hahz^gP52B*(J8m`Vii)A{ZX*v5>nSBlDez-j2wssg+lrU&9UH0rLohAAEuHn>'
'k~TJR8!{rE_FHd6BD^;wMnYJpB7tcJN3{Mp8c^M93?+tyH<?s>UWA{u07OseW7*iDsMhC3<UAA>V-3Mc3*mP$*+|oH%tceL0F85@oxWci@X&$4'
'QsM3<E6wY*O!tT~qCo$4t6vMyaj;3egqewxI3mbc+FMkssnRvLjEOyApc(q$v_2lW@Z*Z02O@W9uKa#l!YJBCXuQu+?6wdI#8wWT=D@QLz2R-('
'{2|Gl?WCNu^8m$KU_>W`Uy-_{NY@a5sdHPm*UpYwLPV2J{?{Q`Pi7$iq&J`$#ToPHfT$Jl_7p(1hDei^ft{GSlQsoF4k!t$0^F9km4Ltzx`~P}'
'3y}?pimELTjn^zhCrb+n{)-YVN(8W#kOIV-Z=?!1%Sru&>>Qb-UTatnd7xaa3;0LKcFQ$aw<7E<S*9D{8O2(Zv8WwPSgIn6k%}aABRs#;iX0+Y'
'ay>@>v`R(RfbfLeZZP0@hwelTqXOK?ZF*Aht<Ywi#d(%OIsfeZq?_9X`q;FyXxi0D@O5;z!~+OI&nWpVfX*ZK-pMLccZm_;P}$UL&YgZQ+?!su'
'<O{>2Cuu^&`9;K2U$*4US;hyHS9z3K6MZ&ogxk7_Rum1XxYoRfWr4K$t&^1pYx>Y(xBCjTCR+*rYmeZz!jokKgKTH83F3L~ndkXwFj`CHIv^YI'
'F35p|veFQdApp6PwaZ)_Vte3X!E8w`I96<+6PU{W)rxL&8wtzrT@v(+VJ8dpV3xqzJ0Q*7$!1jOI1vuKW?#Pa?U3Y987sZLMvZph^lFEPzxnW%'
's$}s8EJheRii00r38mc-U-<$-U2fUO-#>k3Kaxg2a_LL%9lUF*LPG5eW<rXHykXRdJf*0)U5}FNXiTGv$ss^|1z4_3FySOJn;hsDGaPcD0J<cU'
'a59LF?`E}*8@BoFtp@iHSo+HfnJ{I9Im28G{tR{{g8v*JxyFSf{>ZICvZ_{S^HLF@oW|@k%9t<WpCe@#i-mu362KQZf8_o|Diczn<()tf0bGD&'
'>Jd8ZQz0hxQibH&$?)62^%1dpap`0A(#PuA)uM}5Di02cSafyfNyx8Rf70H)UD;?dK<cUil7Ea(bk;xb@f9!3L1vPwrn39eh{NU=e%24{pV;Mq'
'@X9pL?Q@6G^v4kY*uow`wU@&Nl9S|8gM+x{)M;?;97_Z&qG+E9B_{Gil*ukm2Gd<q8w*&!x;vR+#E!0Rapj@-@}*ni)x~1hHb7~Bc}z_I>3ni}'
'UPlZZQ`|X}@0cZDF3=-iEId8Z`NXQd<-On_mI|6=DJ`<(?geM(S95mu62yZ}_^y}u68PL!tR9GyKDq1iel8$|BplmnW3+%CBrQG;+HaHiJ{Y;v'
'ws2ziH7~lckf08ZL^t;`_mHZqF;dmC#I``~bxS6f*c-Pa@6|4iKOn=8Y5_bdKKnhpNlE>I0j-eAc3rK>>D`b(<&Y5U&1r7Lp$69CYNvs9<{NIb'
'zePFnf<uX}C7U>%TE&lT6{3KB9MiMWN58{K`MKB5eum$nyhNa!c}+kN3;p`UQ-EhsvF9i3`9IjK&xU5Ny0j{x#%s!FsQoh#(TEc|cc70D1&qX;'
'Ml@#5;$Oi2{)awAIXPi}12l<VMHtK_l!bqy^mkgrYGxl@!y4*CaHV3)&_F6@vHZU1IRgzNm4X?<dQhDxU*d5{rL40(HUI|${(kKjwUe6<X&y@e'
'XSPtpvK6r+%wW9QP4#0+I@-dnuZIk!Qxrn*6!)BlMNNr=`t?Ktoa2cLwp>{!hJAYf?^rdc${StLn`1(a{P~kuVe%YuX}=Y;+<qkQkQhHu;Auy1'
'SWH*Ih6an7(Y}gnL;>$!ZWGsZRmkT?L)@fFiveBgAm~=2CM{`n>LD9!%QMFW{ew80uE3@bJtcTtI0JVRQ7}h4(U0ROF2Tf|SI66P$Z4Ya<31e;'
'JHl@$k53k1{v0aa=SlwA;cjo&sJNi2yTm@Wr|$S$M7a05<spBhRxcVndMmYOvAk8a>mXUHO6uc?oJtNIx>GeUO+5I7h53(E`__f@O}vo-?!EU9'
'Op7>YyXP$EVq%UbyHoS7E<!nR-kTVB?$Px~UqE}l5Lvc@@zen4nX?oG0wjXE`<$t}s{0lU?!L8VC!JjDdm}<8t(Db#{o#7-yfyfxU&CeK{54PM'
';Nu_|T=rIf0Ov|N=T7{=)D*@oq2kWv15Zo4=_#`743A5Ws6~7Fi;bHJlp_yaBGBRiU7M~q30RXrc7n2!c<W<jeJ7BOVQUO!ci{CXT&41YO~dMu'
'cnUoG-DWfi_fYX}t&~~2-iJ(Y-8o|IU=keGLzMANywDNsnsu`;K1B9I=cCV`_xiqP&H`6D8)e3WI|&~PUJZ@X-6I&uFK1I12u3r;bZh3@t_=JZ'
'sKzIEemvT${7}J8VQ;b}Ca)PD5YW*L(AyHM33<aGuA6OzcYWxzo!H9U-_}`yKO2$7eavu`C~}>)tL+-tbVlKN*bg>p*V?Dp1DIX8Jc{-8{(KQC'
'?aQCOFWPjL|HZYU$2|wn$=pq%=Z<CKT`E}kkG_G;(x$ySMJ;3OS%s@vO2;PdMJ68gb~Z)Epzgplb}cm>LVax%AX3WRw-hgOxgLIMU3+{j(py|x'
'I8>#;m8HNtSF~+B3p^;y{$5Pz0bAdtm@)d~X%nP2SbE6dsle+Eok_y)o?N*VxiQ~^K6pHg`2kIe&Uhoa-n;1dCruqN&F;9yv#NI6!ZwkxFyjco'
'3x!OQiIn=GW2pCCgW$WQ$tZ%pPnHb9_04(mZFfm>JiZNtQOOfPMYHQaX1zFz&U}78^UQJZvyTXAC$9j!`QQ-&5F#8!jP5Gk{&>`#x32H3)i+64'
'f`I}&2giy^%`eAbeSmuEr1~~gkQFnk#}Gtp;}RbT0s|9zfUxt6He-9%5QOc}hzCdvryJdIu5npB2<~V1N679t+2B6JF^9V&ZIA8}0O=S-@nej6'
'#3?}wZ@}=y<6x_I7xfQ!FE9HQ`}M=<b}r}-LF3ba_8>NHNX4VT4}fb;Hm$j}S}q%xb0)@gU5%sq6~o-z&&9jlfZaNOSi$ct&7+lcxzmA{%bW)b'
'V=RRy3SXeQBYiiVvKY&O!xDG&2>;vg`?(ed9BTr`yQeKx0a?A|5p;JR*B%j(6Yu!pes>Pd7DpM5j)EOf1$)UH5Y$i3WJ(g|cvb$uZHeJ|Rk*&k'
'D`ip$4vT~&&Q=ef{1r+&a2tP1=P?o>QY8Lt9y<B|P)h>@6aWAK2mk;8VqGHQnUx+1007z=000&M003cWUt@1=aBp*EUvp(+bZKvHb1rasm04MD'
'+{h7r7oh*p(LRt9Q}kHCCc)SYvT=-Hfvj!B!sbC45ItmZrdK4J=_X}O6#w^B^}%bX!xSKqebjw?-MYBA__@}gT2=wqor0XzTq)iPSu<VAR<p07'
'J;)9h$U9N*nN&>Pf%>XMOP^YHDg%}aUdnw3@%h>L+4X@26lASbfo83!K)1YVG*eK3Vy$E?F40ip66Cm(`7vhK2ca3ydDG$&xNmr!Yg$*GpUFCd'
'h>5BK<kLcVWIHIN0;E^xpiqvO{Sz_}O-nplE^3rh>_yE>FJUI7%5k6tjY58us80^%<sNE4by13jw+GTLlU0Sh8A=29kaakR8!j#`&d&-ZE0(53'
'*P^A;6eVgTnvU1CY>mk@sc%;0yyaQRwFd2%^+V@pZsfrAp%lCRyFwA%ev$^K{NB%I#(jL{chEE9B1MTCsUg)FT8sNFf8`p!z`laIjo9B6Zw?~+'
'T=F_%ze<q-Igvhz(D~V4M#7mt?6ZP~V?2CTa<6!`IX`3gG`M5K<nA~0oE<6o5tQDrQfN2AyIe>+!2ZWR${Or=#hV6Q<(JY=4h3s0qf{eFGGVCT'
'KOF{4WSB*cLBy1l?Zy})LJO!GN$Hgn3%&E@SYxng@n2wtP_*4eG|M|1&qxDT)EFQri3vKEAtT0>?8x6H4|HriO+-b(oG3BI9maAzq(+jkKo&(n'
'0fWvrq7Ku&E>1jpFf2P&&&?W6btGd&>}I>N`{oXY8bcbqgHzqBoZfktB9ZrY^#`S-A|n$eo5~4$_~t01+@ggFwSIx@J2DVgM<fug$u6`tMkb;+'
'hNxR50|uoARzhv`ig%o%(#D`(4fMv(A~mcxJ+;tG^8(9)r^Q~eH&0WWF2__RCy{k6&5GZLZ=%Npef|1swHk@LDB@0ud=CMmQbEMm?A>6ecaM?Y'
'OvP4QpfqELg~3nEl4dmkRzN;b-9h^mFo4^y(pNQ4IGBptZxODFzku&GQXSQP7X>Uc)s4w4>zbpj+q{CF!<ez*bmM2X7%gNrAA&B=^HCJ;I@Ghp'
'fpxqd;?fg{`Uuq+?vRIueoTWpT#ZfX5<=?HP>yC1?a>Jlw{TBk6a0G};2&RO;y!Qeo9S*kMxmg-Ls$HqzW?;g=TBcgq}SJ9(qBIP?ZZEUhy}qo'
'Ix>!B7uisKVy{`y+c3Z{SK(uQ;8#EXA@owkP-obFP>c8g_qo`kEaB>AbG6>i2*=eew$hL+hi*=;Rfh!v?y-WV^45ILMIi%QnNXNszjVEhv2O&E'
'Fd@g<J+aVfAxe~J7Ad*`$PP%-1heUxSdN-w*frQPAbA*A=wQQcCcPyjt0%qvBX%dtu7YIA`ZTfHew$SZsQIomD)~Jl5psZLZi3OK5qa8#G=icl'
'%V4^A007A&6*YA9B4>=q+HBN?rWBoFd%~O3Qni$SABWs?QfO9i^hy5KC@&~?=4rXZbi5T!1Nk@+39=cK5X%PZkZx6I-Mqr!Rl8utxTg8JuOX8J'
'x)PeBB&Z8)QGm&Yq2z|l2&olLD~{j$j>T_YP;8czwH*?bgsQ>BH85hV)er1n2U%i!H$q`Rc2phpfl_~`0D*x4qz%vDEqZ_<97Xt*`w;dpARZ|C'
'A=H?lSBwa<3`4KVD;8h<xMIKWL$7EwSz`1#YLT{-y-Iqb!<?^tlhSC~nhPZiEx&8kV-H91&yRwT)UHTJ@~+oo<1ZM#7+JpMw`j%(OJx&Z72ot*'
'YfkJjI#UKsBUUd&YwCWm+;ZJh1P?D1)Uq1&8aqv{wj3<@HyJsvI`fqGOQ&r%8pek-Mrt=(0#_3>8re7_N2q*QGpx%xh3y9?3}BqKDuF!isY5AV'
'?;(7o=@VR;w5wv+=woA+Gd4IH19nGI+DDSvznzu6MO(zEC!UhR(QzQO;336BnQD`D<iG^N%RKSSqElzzBGY8l-kj;uJs!7+3Z|fzb%KE~K}+A#'
'VasY>sZZVH<$b!5SkbL-o^LBNu10LnC$vWPq#mOYje6;c8b7f2cx-ibSt$#X2(*_7BwQ-d0T)b^5ghe(E62*WuMtFM_!&>BB2Xwz*J45Zn-q1}'
'>mTc;6b3N-X^X%9cr}Xz6U$z~(a|roE<8wi8$1u=9ST1Ugl8dpknys_fne?^iZJ{B>WM%z&NkYcGhhZ3v`LWRZ!|uJT`elJHwko!kSgsO!|7N;'
'67Y=H1q)<f5T7XF9bq&~1_gz-5jPo{yWW)%Z|sMFOdM>+X%cu&!Dhl3C+s26CStI0R(i6`vVIc~pSdvuvYq2&G9|D%N<kiM2nf*IY7Vnpkz4Fu'
'eYJY&p~j!OT=kx;T`XLq0_WDnLq%ej^Fd>+?<swL|6(6`p+7BebmmwHwCI9p{Tkz;Q;ilJ98h!dH0eFEBZdOKjXb;@G*5=b_qdxL)9NO(cHUgq'
'2*?l;T1_dT$L^<l7%EMeq8K!ot#`a+ymfagBm7D|$J`^9{D@Eh@C~Dj-_kD0e*HAu!q7vmAZ(Y6Zj<Pt7Mzu$p+@R&mrk%aEx566+_K+crE;!Z'
'faM5JT2Xf}YZnpQ8H9N6wYDTjKHDO&uCPLrcgM5jjP2?1V0Xz8*Tw6rB^f%UZ1#kf(A3}CSN!sny=OXKyIWYha1Vnbc@vMXO6Y2X?{w%yx>;{8'
'-LjiE+Z93mDc&4EgUdp{XT6O)r8i4bQ0sVv+OmL;ET-oupv?WA=OlDeO-kPL>}D)R8G^mK4S73GjPH-sWQbJbh*L+hcM0XzyOV%hI&j+C$Hgc?'
'c#32$7dSj~l=z8S3*Diz2i_v0M_Xaj*Vru$05VmCF8@nW!zsFt_ZFBK(2YbEFBe>q<56Hs;kT7FCGzw)m!8ZMZIia}XlJ(A@<se?GSeGXF*1X}'
'O*1J8ET!i5C>aI~*zu!7P3q@$ai3yMtJ^1)<EWVp7FvM;v!LCG5=DqvXKotv<95cB{@Ku`4#3)@3_(+Ly%JywafuKJeR43(4XZI70_TnzPDhZZ'
'_FkGunYZt8k=a0MI2xxuuq3S)?7Q)U72uKUa<6POm9Rg>x!l5F`#(@i0|XQR000O8003fL5WAGa9s>XXa0UPX6951JVQF7uZ*6dIb7fz1Z*^{D'
'E^v8;RZVZ>HW0lB$bT59C_t*REGtenNbAF5(L;gV><9K%6b3DgV!{$BkW`$s=zs4FC41A2ipqv$lEe4ry`jcQlDxN-YQaC7ut%@|#2SKA#_OQ1'
'foebx%G6+c9#f#|b{b6pUs;Dg!_E$-rafQYe#ny~xx8qc?O?lYhA=qXZb5gwb%8L94T>@R<;67Dsjxo}h_klcdFX>1s&F12)gw;WJ?(1s?gacP'
'g7Bxt=7kKwad&Y!gM&8p9b;xt4=$#02Tf~Lkaw3CKo{ly&Y=nj*69b$N<Y2(0BWeU1#Q3@TzIT3SyAl=bMQIbTd3{P5DFXMRoA!ZM-v#4`SV^m'
'GAr*6wRU10_{~>0U<UZ4ec0MV9LjMIrA8~xJirFZbx{=A<V*e~KE&$H6nWHPA6?09k4BKRC|G8Y9CwKX<-rc2AHt-MuW=&x7$_6fc_CIAl-a~e'
'T*>{%`(G1!Qmvq}UGLC)tP?4}fdndJOX+D7#K}uYBgTuU#yYS~LsV=YX?Q?E1ab37C|@MTS9Fzdch3QpbLv^he+Mw6`Gz*4XV7UQCfJGarxE{q'
'f|xuJ7+A)D*Q;nAu_Zexh*8O$u*{26lDuhrR=fz3mTu*15Z%Pn2Ad5e-QabV%;dq>5Ab==tiZSS*glg1q)SujdsXM~b+0|pS3&P=h0NQ+4(klK'
'HV<f!6M0049PswHe4fQw8$1aTtE(cvB?+rlk<<OnI=|u3`X;}<nvrqZOIstv?_BS&Tub;7S~_rXQvNZ&@oZ$ef&7Z?<9ET6hw(?RX-Wh_T1!Gq'
'({di>QMCgX!w;P}Bb4cck#Y*DOM{F@!|~DJ8m;p3n(x<AF0W>a+@2~#@qZ;%tE&Sa`QG;Ypsu2-ko^50MLVT5la9D4iZm>)`A$j6aRHxtu?T5d'
'%y^8g$N%A>R*(-ylfX_aSe%@3GY!krxnwgbB459TjV~DG6VTLRY_HUws)I;=*^_dvZ>OG^WjyRLh4fTP$;4$zElJAnq)ct)=ybq_`U-jxxlTO|'
'>7T&Fc6fR54^T@31QY-O00;m80AgJ|67Qit1ONc#2mk;S0001CX<uV+ZE$aMWnXq_b7^mGE^v8`Ro_qJI1qkE;(xIE;uM7(pk;5>QyzeYy-F-_'
'yTD6TS#C0EEfU*jI|UZ;zu(wS+Q9wTix5pF<C$;1`DQX2jZRf7ODKtu``V}itR;Uuf;`HO_wO&!T7dysKuT)Gy`Kfm4R=sKlgg-uxD}967$}@F'
'*=RJ{-3eyKs@Amz7$c(5$~eNMRE|5Lq}|=w-RZ2$Vsop4eRrpXij-grBNo6a!yB-Pkk*;D@TXG}ZR5%A4#5j^PNmfdfV7%8ML;Z9ju_x|=lL5x'
')W9u}FE@AOF}$~A0hKa996ULqYeXmHGV}R;&^rs3+!`(|IyGP{CAwYIq8O0#%Sr3hP2@}&a&j}Z8M#tKb5RnIrB*_^G{!T~C<mlaB@lRmqK%cT'
'MF|Rv^>#y6P-`$L!C+Ne*ZT&ns)Hw@ls<=bym-hqb;p~IKIFexMN8<w&GfQ=HEsA32Cv+pu-@lQF)29}&YsBf^x1(uQ?e4vG?vbe+5P2(GH5J?'
'$Hwa~Ai;#r_VhJaPtA~q9J`u<TnnS*jGj!{+4a|(>+k36e)@fSeZ_viU|+94pI_2>h7A@v8Fw{R1l1C2%8It+nXrtnxv2S~hNSO(pq2~l?6+1r'
'(ZIPe$|Uqs71;f?u%5dBhDd<Y5>C$UKA(6U(6^bQQhpNls_#|V`I8MKf^_<XE#@__7Vy|7&>My1Z%NL4FPJuf?GO^mq-QN9jgU->C#Y?Hc=Ty6'
'{GpXU_12@~kI~v!;X7<o6qrP*I7~I0r)2g*-LnQdp<d-QjX#sYFul$@S$NI)l5pa?+Gdka^8p`^!mXl^GNnN#!D&V{cPly{6K*4h!HF%djB7kn'
'CgCYVjB7A1q2Ioo((!nEpaY*FNbmhG=&R8!c7(ii0DODse3BhiuXgmBlFgxzjK0pt14$w;Lh2T$ABEYv?(Ry-ZaSQK9u&|Y`MK`@qm?C!p#au!'
'ol}nOr|DaUg2Y+rLFZBK6>eI0_%S8iImvK5YiN*`kSM3N)ck#nKU<D6{SiD^5vDc&`v50?gELgthlOzYt!Z(oFXSRTvJip=6jW9p-e>q<*Ctq@'
'8l)Efsj6YWQDoN3!>jd$8ddwYwH6Hay3+!kbT_?F17&_?=i9;Z@~WR%vEo<(AYb@-O8Vd_GW-Bi{DAy|*A{E$WyzMC)?wEWZRBHsZ6lLSmc*yc'
'XZ}BqRup#L&z;7g`2FZuMIE$8vbeNuPPL=s%&Dk~EGlCBZnPpxo9`|2zHj!NzqXo<p}I04+1rbN<C2i@$U@aN|MlPiHK>^7UPqCFLF$NfK;?ov'
'FUPRuk}lnBlE&yW8h%mN)-UW8*D%{ZoKKJs`ye#)Utz)PK6jvWcF?aFfg@87=gUu5mm(z_{JyYsi>Rl1du&<Y3aM-S;3tDh2A>L)fI~U(Vg=sg'
'FHlPZ1QY-O00;m80AgL+f{BNz0{{R62mk;R0001OWn^h#UvzR|ZgXRDX<}tAaCwDQYj5H>6#b6Gf4J)}ldM8!Dz(xGDJ!9pog%dZYGFnzR9S`?'
'@NQy9wgY8n```E42~A)hMF`^So8xot%Y=}r<O8!)7B<XSt_O#mFeTS;Q2Gz9W0+V2l3B6mQ1C+OBWP|bEe%)&cE@4*1WE#b5jG}-^m`kv3ZQgT'
'p%<qVL{Tbjfk~+>;wWkQy?(E;{%(|fTPXAPXpW{qDw*BoVs(O<;nKa|?`3=gRx@eVTC8}$s!XV%-;W??a$B)2AA+$O{(`Sc@(6y8faSS5P+pYw'
'IE0l_IVA8a%MA}<@C)R^*kvZx_IcRv0e(D#IxzGUn1;yrdikI-o;y0uy>JUe?Cu5imis6ZZ@J}iACQN1JQ>m3`^k)czneVG(ntC*9jEss0%C<c'
'k}yW*q6|V%x-ms9QpWdU&6EEJ9H-wVx2Z#1m(>NJvvly*swgYVsXdnOq25j(W{Y`B=fBN7wvhJo1-LhL*{Xxg70akr%Dx9by1h&3c=G75KiZ=F'
'cwUzmNBx^eS65k4yVIOyCMnoUa1{ZP3*MnHThRI4V>%i?ITBYS48snltO{B;he@s)dszP&zAd=4CIX68;Hx3-lxT~A66%5t+geRd4^CKJuq)0A'
'YQ&!$4{{BeQ7Z~=EGtTr%yq>(wDuU^jmR+&??m|9XzZ&DU&ml#J8+G`6cT>aHZK`}j@-CECCTeqrN~UUE~fY`HP)f347L2lCccj%u-IK(GuQpe'
'r})!(gq#S2v01i=KZr9D->ShOb_<`{Hq*!gHEWAL8>}OXUViO1xXe0cTy;|gq||D~Rw5Vn*m3JvQTYbWEFf^;;XHIH0FILU(`$>mL3T~1UD!@Z'
'_x3}_64zqUx_oZm#^e4|C=SUC8gXYdYCEdyT5HS(8eyHiKGM0nRd?F83^Wrc>amipm{P5k4#>suuSYz0C>|2px+6^{YnS@Kb|4o_$iUB1$8h+@'
'E$1@8nI1E}-G`o=E4b_E=GBEDINweZxbYfJrsefB{FK0l!N4mTctsz=e`tC-qKYELt0*9q@$<ht+8tg`g92}_OfcHEBa9Z~$rPrGInFb<AAS9@'
'7=1}$d3MP2Nv`kK2vlWpP_|8rrTj+nYIxlw!(Bo_?Ha}hE!<?q4hh_C7iSelG3bm1?|nH9+9A+MVsaLzB)P%C1YE9FCge6DmE8<JJHq~g-QXoG'
'S>di=3@D-0<%UwyM6J$LzxOv#O9KQH000080001DU41laZQ}|60O21101^NI0BvPtX<=V-Wpi(Ja${w4E^v9ZS#3|_Iu`zp#DB1|U(&g4X&Ejv'
'3)7L9nKB^3&@C`?C4?e3aa#{boQ)k`)aAeLIkxlK6j-fpCDg{TkI(Dl=NQLvzKARi$nkNMumy=&B+@4$Vj?3NhGZ$El7}Q=X~Y#_k4eaV__5@R'
'WiMq?WU0@{Z!wd&L?ns@Jsz7?eaCUOx8|vci092Sm8Hz{2#*qxDnjE}D5|)K<@VP0Rz519VxNoR*%z^5kLsSLvKSNP&)-tS#e-aCiihQ6HcL|B'
'Gg(eoEK24)WaR*lnAI?$Y8mobuHFP5Q7`>&PNkv=&)=O~T}{T1K1)!0oaCLHem^0js<!>F=J!vUx=zFbPCay)m%B_lT@tW4%|bOgrlDjV6tuks'
'W28ROV=<kDEO0xd_nFLu2oJWm2z*io9*v3KyuV<|t%$g%ArGixjZIxE>p;uV0)L2@S%G{`?0bk#0DeZPEWUT0^YORSBk$<!?BbVq{p0NP{Pb%4'
'&6|!dFRrIw$CqG>Ge;wwPCpb6ECpR5x&#uzb%yv>r5VHTCosHYx6?LIE?M27?%v=%%si<jSR?{QxFi9WMnOn6wS)o5(tDOV71L!xA7anATv2)F'
'`XUwGhEuHrtR^;YjR^W8dGhQ4`nEE&pJuUZ950z_3h&h7V^g~Crwjv9QOOD$+3$)CUMi(Gdr}awghRA=taPy8+p4V6r-~aNxLQK-vDlqEJDBx5'
'D_Hlu18J)=Of?q!BH;02<WT8z?sPycA{D3<Rm9+DE>aTmm|^f*C)iBEIen0yDv7(h&d+Fid3jKaZuMim0T#GsDF6w$^=5e6A@9g=u)Am0F`Wm*'
'y|^5wsYtuzCk->Ee^*5(6zjSocu?`4zeq)vcnL&A0_d2^QCSkQ8`!^#O5BqFR8t?2YHH+QFeoUPl1$PC)0Gf%#6C#5GahG;#E2*Ny99i7KQu76'
'9+XrFOesVX*V&Oza%V1g%vz^YPK)RxkFp46x_ZS9aSR6c;SCfNgdp%t4uZc^p17685Qt*^2o}PMhX1X5(AM%rCZU8zIvCAO{VYvctm+$AcEpsD'
'ur}a+J(9J|X-dUaO<Qmo&X+a$R$s8h9dwPNJ*}wTSmT_-W9ln@&pZTkgq4BQ>2!2lUF(j{|CCMjz`YW@9b;29hV?kMxmBl-$755~8@=0z>E{s{'
'l24z={-C9*S|j^PbahU@O^v@h&~FX40-6wdn?i0qRf>IM#Mc~{gGb8#opB&4#vX48hJc9?YP4Vn2#H<t4)*0S?(0P}jn3ew;fMX^feOHI6hvG~'
'IM#^2Wd0qQG4LLs3JGb7=n{s)*DUCe#0*X{f{+V5i0EbU!eS4gbr$QNEGY4YZ{`XI&5AP5IV6I<>nK?WEoQ+1TQSGJUE>AQ@xQK5r(^Hy_vyv;'
'q>T(fqO8`5z;iPmnhZ>{7-^Y~7#wFS?fB<|9Ca-}HEzjbdQW+XHXnkW^Gv%ecSp}Uj68_+&RN44^NdN?wc<vF<@wQ~=>?2qhQ8l52(xCtu^Z{o'
'c=~hfoqs_Chr1s!L-pCV_wmmU`$nYA8!K{uZ-dD5(;wdD)zS3II~gB+n_gU;8?lgnZkc+WmJRgLjK=ixddjW%ly6>fdo;t$t*+k?hz3b0!@^^k'
'&F9b)z@d=`$0oUE2N)2ix!N_wJX&cLcXmLJXL8QJ*g%X{#)RsNH5t~dV6@5&tWLLO9C-7T!9Ka$%PF0Tz&7H1&jThrlSnQ$2M78v(8Yhp<3K|J'
'IM2}&otDy{vfs?d9L;0~b^c^aK43t_z-k^RnIiL)Moi`&Gd<YOFJQLNaAEm{G>ZW_{?DQpGfZOBAoU-X+=oMLKG)+;S@uZT_1GFgucbo164lGy'
'tj`3^xe3h?&nbAYgf0j7FFG>KpUX{dv^<D$?;cBs;WqfACs_zC^b8M09_!p644iHy6=2R%Jb(*;8R@3VF_T9RJb-8|8MsPmSz?Y~*fH8V!k~6;'
'E5f7-M5aa+H7BQ&F-{IZrRw;%@y{REXJ=jAK;oMBaC~WiH_e<CTDsQwpl_mr@qfj|Uqu!MWk77Q=v0FPIbtz7SmqwK{s9-Gd4Re!kgna*7}YUn'
'Y3azD+j`vXn^)AjG~Jlx$P!jsom5Kw0%DEpiSfN9B3Ay*;P%GR@10vq*s5=AxN_Z~x6=3lRc2xF=r8Z|{OJ4Go1FeNKD%^KeXK^!et86t<PZh{'
's4e(0xxUI5SQIX}8lv#93Bq<?NSEEghxhvf%{2;lY4ypc5g8!=w^rtl=H<J5_;bG1%Z_bg+o=xs3N3pkD=qW+ab;vaYl_1HX)b7bc-4}FN-yD$'
'EL_LRq9*Oxu(Db`A7WABh=Ld{S>sv<MYqGGr?MEp)fZ6&4U%iW4?w(BHxSWV(-T+j8HzVW&#jk02=GlVq4YAK3}T^^UnFQIZf8NRbowy>=NjPU'
'V$JwJ$kh_g1~L!nLXs$xSWe7mfh{PBW@9p^BD5|{b;AcFQ^C(gN|U8#4=k1a8vIsvM(D#G%ymakEs^|cDC!wGUds^Z3GL`(@v6y9Q7A`=DUBD*'
'4WYHqT?9w^u&v7m&2s<nTPz5z6q-{ElRwB%zXCG0p-*qlMW$_gDX6Q;qI}};Ku^1IdXQ9}CbSk{_xDP2(&@Hur$23u8ouPM;(cx4wss*<WJ1C0'
'v;11<rL?t(u30_M5jta`KI(N6SB`+MS|wnY6Ejer`<EKfh@D76vCw%IaCxVrA(cpg;Qto0{MhG^iMouJ_zTEh!9)9~Jj#VcE+kGbFHKktBi84R'
'BorAJMWukxV$o2t$S9<b0<a)Ung*7h8G#xI4q+1Hb_|$-yZRLGe0UfV;rm$%>Zu3DVk=f6z5(h3;FHOdT8cO)D*eKC^8*V6JB;ehWWDCw@xy>d'
'{a|Vo+otwR2P!8VczKouuyJ{E{5m<koQ$XXKmh+Co7N~}?+L+8(oZ?iA*{Jv&+18Xe0)9`f3L1DH_N*RJWT$%-#kr9b?`v>bHA-E_!~M6eJ7aT'
'c=xXu-d)r5=@_0|ef<Yn-hFWbke=#|o6Wu&kDKUA$0B@|L21dBHNju~_9#{Qb^`0H9|;Jth`I09x-$wa2By?+HrDTGfXIeLRrC`PhS+MZwX79<'
'Y;FornrrmAsQgX$msTt2eRQ(r=P;~Y*`?1i7vi)FnSqqMR?S*cy|%k{eDzv5x0(l?_9e(VVb$j{LpR8N&AKw&8hLDGxhs?o{gh3*wWjKgw&Ujw'
'w-PPZ6|1mR^9P^?U~=&UM^n)NzhHiQ8@$m;?h9ubAr&7Y)F<FVi}Up>4EDIBKuM65z2G(@uxOa8cDD4*Rl{HCp4T^cxtlgz4AXCvhyCrMX~NCQ'
'O8OdPgXDDqxZ0cY=4V49;AcZ>>1vlDV~IQH57%#5O*~uDA}=n5|MZNhYR$rRJZdg~pj^2{hu9**-c?rc^vcX0KPK2!zasyRBjUY5hsK-+k$Y{)'
'99$5FO=EQdz7cs}Q+U(P;EBFu?9Lrj2H`-lgI%)QHg0c|@mVpg84%|E_aFD_|7KXflxR}A@?0yfEsFlVtCd6Y>2}Os=6C&nGp!lc(*Idx`GYuz'
'cx$IyF^dv&zJPkx43Y-Fv4z2x*ppzJuRBoe($@cjVl3R=`Y%vR0|XQR000O8003fLYf*tNss;c69~%Gw5&!@IZDnL>VP9%*VqbG~Z*pZWaCy~P'
'U2oeq6n!_~|G@GkR%bhIkpY7@4NF^hOV`wa;{Y3iK#*-JQ6q~QMYS<J|My*fNTMXlb=si7^bpzP<t2H};k}nmz~~xzUXbyO5zj;6I-!gsKaMH)'
'c}U~5+v#?6D@}`!JY17*ClI6gCJE!)eB^1oF-9=*PexKMopd|r(>D|3f}7zj;Uxal!{?JXXQ$q~=_~JYKAlZGVS)PyGqR?fU}u<e7A8Gs7}3v!'
'^_*_!_4LDhHhu41yg&Ug@m`%?PQ0tx`>I#Fjqq!DWa<Mx#^ANSj?;@X@6XBK`xmQ?s(!nj`PunodNubhC$Nq;m*8XVKlEM>M|~7T)aSiDE^A|j'
'>z}aWfY;e%K0BM_-d^<4I*fZKeH6xf9qM7i(M#dTQA{i&KTXj&S%v;Py13+&k#V<!;8~FXc@bUuk(Y|*UP_|C(VgOvoQHA1{9!^^8m2JulH~Nu'
'G00Dm;i4SN2*@n1_8HSE*1aW-gYZ^Oj0V3Vo+S~v27LSAp4}L02?GQ=MRvXoJ2|%>rUcEhSV(um7-eljL18N0s&&A*SP6;L=4=te0h6|AZSSla'
'Is1b9ZL6uFE6-xDbSe<;mtWU}-_g}rfNZoTKEJECJvTmoOlj;b=xQ@Yt8mE`PY?xe(7)&djR`1Lyo?ck1gF_M8J&#Lg3<^K-uh9RTS&_DxG|iO'
'G=U&Qv@-+Vg%3l}J=wr`v5{d-J_dLh6O#mZE5_`$Vn;h%cHa?y1xf~!ZIwOz*FXY00AL5=q>sgX{hHC~a!C>neb`TuC|t_0c`RVThUu=NpaQg<'
'_1fl)WnJUfG=?+A2lGur@C_`pIZ;luc;IMrj|i=e(d0p_4a}-!7^j&s<a18e35P=sdW~;MPX`UBwjj<nwHEQPoFTVlttmD0Q9A3}JAtc~9*u{q'
'-z`=G4!3}}#Ymh+nsK)lZY|bMvkB-c6by-SRUiN<z~@;C0$c$aE)oiFncs?J{;UZ1!${((9gF=y%7P%@!>Sis7A77d2`_m+WSXiXYFpJ-pp7lQ'
'itl6kIR*kDZ=jEKA<|0T2#jY$HtzOJs^eJul$MCu?hT*kE76IE#H)ltO4Rd^6%d57m^>t8$>Ey}I=I~-#E1fPP9FI6_{C^6y4kf1v8%(XhhNZ-'
'qfrYm#Q?c^56+~ghW1)m)+pxVnnq;|0DcbuGiV@L0Qkpzeu1{OJ?&bdd<2ZO9cuj=@n3!@;~10{U}FKYvSlzaL!x-ZvDJa@2lN8?&{59fN}x4N'
'BSMniXn2C2qn<HRl~4r}O}W~zl$$^SoFoDXc3ZA>haEFXCmHu(Tzig&PA_ca==(5kv?-#R*?a>vC>TSAZf0g6EsJ<dr%oy0&zjj~-maxFz+Bb*'
'kh5gbhiho7#BG+F&ajC=b@P@eE%t9|RuEV^GzzYMTPze3u6kMeB^a*!TkEX}TAiaNB+uO9mh<JBu{nPLaW~BpQFFr?wOym!t@!(l?S4i@_&btF'
'A0ap+M{4pT+*o~8dqYdzC?0%qHOIBB2lyCbFJZAC9m^as0LIeet>SH0H!rfZyKQa|3r<DBWv3P^Zt=LCWUSc*n|7YDVi;^%sm79Ju&}YcGrm_&'
'GURHhVtIbeV=K>zwNie}!O<DOc7F1i)tt`%Q+h^KxmSh1Prg-!MJ?yfSw^aK8pst3f`Z*rtXu0ES>QLyxJodGZegWLJ5f9}?|i}v77(Uk3|fb~'
'wqUj|CFCFi7biVysXbYWc_fzarULs$z9<tyRgxB!j%|S<WN_h%XtoL>A3mxXodbBp7t_nRs2{ClNk7MW2X)4*j}c6;+AEw`oiM7w#1a^GP1I|o'
'T}yIqnkWn6zV3`a1&C7qZHiYQmb$M={m?yE)W1h>%g7Jc8Vk4(5T+G&7vA1^OF!`!5vktzzZ*;kXicvE(Mps11RcpYOknO6n$53Y)>tJ7)X{}o'
'FXId1ThvyCo6~y|mnsx^e-)ZKc-c-<eb;hI_%7eOCnBWSFxZuli$6^u#U^`JH2o<2M6hE4GzjA`y*nJuHffl3c2PiEg6VfB;*hGTQU5#@dkWoe'
'qE5LGKZ9D>^Op)ZxLnE=d?Cw~lu9njJIqrvQcW}!ex3?iz4bx9^+yIRI}KI_l$;9me-ok!m+I%cl8`-+?YR;@stDCLYV{M7o-VbQp30~EU5?)o'
'^txD+_+gzh8V!=jhr4US=)(r<e}|}Rx?BB+h+zCbP)h*<6aW+e000O8003fLuvH}5y$%2XbTR+{4FCWD0000000000q=5hc003cWUt@1=aBp*E'
'E^v8JO928D02BZK00;m80AgLT&w>$e1^@v54gdfZ00000000000001_f#?nZ0AXogV{dJ6Z*yf|V{dJ3Z*FrgaCuNm0Rj{N6aWAK2mk;8VqGnQ'
'V?7Qe003lt000yK0000000000005+ciWL9=VQF7uZ*6dIb7fy;WNCD7axQRrP)h*<6aW+e000O8003fL*e0S0y=VXcXOIB^6aWAK0000000000'
'q=Cvg003cWUt@1=aBp*EUuAA*X>MgMaCuNm0Rj{N6aWAK2mk;8VqIbtxUKCD000g+000mG0000000000005+cxuO67VQF7uZ*6dIb7fy?axQRr'
'P)h*<6aW+e000O8003fL6iyTy+zS8z%O3y$6aWAK0000000000q=D42003cWUt@1=aBp*EUu$7!XKZCIaCuNm0Rj{N6aWAK2mk;8VqLy@;oh?X'
'002J)000*N0000000000005+c;=KR>VQF7uZ*6dIb7fy_Z)0I>Utw@?E^v8JO928D02BZK00;m80AgM2o^c*71ONcx2><{Y00000000000001_'
'fy%%D0AXogV{dJ6Z*yf|Y;R*>Y+rM9Z*pZWaCuNm0Rj{N6aWAK2mk;8VqGzbelf!g000#t000yK0000000000005+cFU0@=VQF7uZ*6dIb7fy`'
'Z**yIZZ2?nP)h*<6aW+e000O8003fL+wYr6>k9w?r6m9W6aWAK0000000000q=6*T003cWUt@1=aBp*EUvO`1X=8aVaCuNm0Rj{N6aWAK2mk;8'
'VqK?d?E4l9005yT000#L0000000000005+cLEZoWVQF7uZ*6dIb7fy~bzy92ba^gtc~DCM0u%rg000080001DT^r#l8D$Ot0DLz902lxO00000'
'000000HlG6=l}p=X<uV+ZE$aMWnXe-W@&C^ZDnqBE^v8JO928D02BZK00;m80AgKrRY@6H4FCXTF#rG*00000000000001_fhG3<0AXogV{dJ6'
'Z*yf|b7N>_ZDB5Oc~DCM0u%rg000080001DT_WO{l^zKI0NNS=02TlM00000000000HlGi1OWhHX<uV+ZE$aMWnXh;V{~b6ZgVbhc~DCM0u%rg'
'000080001DT@bsJ!yW?w0B{BX022TJ00000000000HlEe4gmmRX<uV+ZE$aMWnXh|b#7!XaCuNm0Rj{N6aWAK2mk;8VqHBF@1Z^f0089(000yK'
'0000000000005+cQ4s+EVQF7uZ*6dIb7fz4X>)0BZZ2?nP)h*<6aW+e000O8003fL+k%OQsRIB20|)>B6951J0000000000q=CT|0RU}fWNBev'
'baG*Cb7OL8Vr4FHc~DCM0u%rg000080001DU41laZQ}|60O21101^NI00000000000HlGN836!oWn^h#Uvg!0Z*_8GWpgfYc~DCM0u%rg00008'
'0001DU29Q+E~*9q03RCw01^NI00000000000HlGgBmn?zWn^h#Uuth+UvqSCa%C=Xc~DCQ1^@s6022Tc0ObS#0EjCA0000'
)
BUNDLE_SHA256 = '475eb9796f45b845392eae74a3f46a99e59449bf360d1890b67468698ebd0ec9'
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
