~currentPath = thisProcess.nowExecutingPath.dirname;
(~folder = PathName.new(~currentPath++"/segments/movil_18102022"));
~folder.entries.do({
	arg path;
	~dataBase = ~dataBase.add(Buffer.read(s, path.fullPath));
});

~currentPath = thisProcess.nowExecutingPath.dirname;
(~folder = PathName.new(~currentPath++"/segments/ataques"));
~folder.entries.do({
	arg path;
	~ataques = ~ataques.add(Buffer.read(s, path.fullPath));
});

~currentPath = thisProcess.nowExecutingPath.dirname;
(~folder = PathName.new(~currentPath++"/segments/cress"));
~folder.entries.do({
	arg path;
	~cress = ~cress.add(Buffer.read(s, path.fullPath));
});

~currentPath = thisProcess.nowExecutingPath.dirname;
(~folder = PathName.new(~currentPath++"/segments/largos"));
~folder.entries.do({
	arg path;
	~largos = ~largos.add(Buffer.read(s, path.fullPath));
});

~currentPath = thisProcess.nowExecutingPath.dirname;
(~folder = PathName.new(~currentPath++"/segments/tremolando"));
~folder.entries.do({
	arg path;
	~tremolando = ~tremolando.add(Buffer.read(s, path.fullPath));
});

~currentPath = thisProcess.nowExecutingPath.dirname;
(~folder = PathName.new(~currentPath++"/segments/electronic"));
~folder.entries.do({
	arg path;
	~electronic = ~electronic.add(Buffer.read(s, path.fullPath));
});

SynthDef(\ataques, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(1, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.4, 0.4, 0 ], [1, 8, 2]));
	sig = sig * env;
	Out.ar(out, sig);
}).add;

SynthDef(\cress, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(1, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.3, 0.9, 0 ], [1, 2, 0.3]));
	sig = sig * env;
	Out.ar(out, sig);
}).add;

SynthDef(\largos, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start, pan = 0;
	var sig, env;
	sig = PlayBuf.ar(1, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.1, 0.15, 0], [0.1, 1, 0.5]));
	sig = sig * env;
	Out.ar(out, Pan2.ar(GVerb.ar(sig, 30, 50), pan));
}).add;

SynthDef(\tremolando, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env, rev;
	sig = PlayBuf.ar(1, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.3, 0.5, 0 ], [2, 10, 0.2]));
	sig = sig * env;
	Out.ar(out, GVerb.ar(sig, 30, 10));
}).add;

SynthDef(\electronic, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env, rev;
	sig = PlayBuf.ar(1, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.3, 0.9, 0 ], [0.5, 3, 0.2]));
	sig = sig * env;
	Out.ar(out, GVerb.ar(sig, 30, 2));
}).add;

SynthDef(\clase4, {
	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(1, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.2, 0.2, 0 ], [2, 30, 2]));
	sig = sig * env;
	Out.ar(out, sig);
}).add;

SynthDef(\granos, {arg bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, attack=0.01;
	var grain;
	grain= PlayBuf.ar(1,bufnum,
		BufRateScale.kr(bufnum),
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(attack,dur),doneAction:2)-0.001);
	Out.ar(0,Pan2.ar(grain, pan))}).add;

SynthDef(\sinegrain, {
	arg pan, freq, amp;
	var grain, sig, out, rev;
	sig = SinOsc.ar(freq, 0, amp);
	grain = (EnvGen.kr(Env.sine(0.01, 0.2), doneAction: 2));
	rev = GVerb.ar(grain*sig, 45, 3, 0.8, 0.5, 50, 0.5, 0.8, mul:2);
	out = Out.ar(0, Pan2.ar(rev, pan))}).add;

SynthDef(\sinegrain2, {
	arg pan, freq, amp;
	var grain, sig, out, rev;
	sig = SinOsc.ar(freq, 0, amp);
	grain = (EnvGen.kr(Env.sine(0.01, 0.25), doneAction: 2));
	rev = GVerb.ar(grain*sig, 50, 30, 0.3, 0.5, 60, 0.5, 0.8, mul:1);
	out = Out.ar(0, Pan2.ar(rev, pan))}).add;

SynthDef(\sin_short_env, {
	arg pan, freq, amp, atack = 0.01;
	var env, sig, out, rev;
	sig = SinOsc.ar(freq, 0, amp, SinOsc.ar(freq, 0.5, 0.2));
	env = (EnvGen.kr(Env.new([0, 0.3, 0.3, 0], [0.3, 0.2, 2],[-5, 0, -5]), doneAction: 2));
	rev = GVerb.ar(env*sig, 40, 3, 0.1, 0.5, 15, 0.8, 0.8, mul:0.1);
	out = Out.ar(0, Pan2.ar(rev, pan))}).add;

SynthDef(\sfgrain, {arg bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, fil_freq=440;
	var grain, rev, filter;
	grain= PlayBuf.ar(1,bufnum,
		BufRateScale.kr(bufnum),
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(0.01, dur),doneAction:2));
	rev = FreeVerb.ar(grain, 0.63, 0.5, 0.5, 0.8);
	filter = Resonz.ar(rev, fil_freq);
	Out.ar(0,Pan2.ar(filter, pan))}).add;

SynthDef(\sfgrain2, {
	|bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, atack = 0.5|
	var grain, rev;
	grain= PlayBuf.ar(1,bufnum,
		BufRateScale.kr(bufnum),
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(atack, dur),doneAction:2)-0.001);
	rev = FreeVerb.ar(grain, 0.01, 0.5, 0.5, 0.2);
	Out.ar(0,Pan2.ar(rev, pan)*0.3)}).add;

SynthDef(\buf, {arg bufnum=0, pan=0.0, env, rate=1, times=[1,1], a=0, s=0.3, d=0.8, r=0;
	var sound, rev, filter;
	sound=PlayBuf.ar(1, bufnum, rate, 1, doneAction: 1);
	env = EnvGen.kr(Env.new([a, s, d, r], times));
	rev = FreeVerb.ar(sound*env, 0.13, 0.5, 0.5, 0.8);
	//filter = Resonz.ar(rev, fil_freq);
	Out.ar(0,Pan2.ar(rev, pan))}).add;

SynthDef(\sfgrainBailey, {arg bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, fil_freq=440;
	var grain, rev, filter;
	grain= Normalizer.ar(PlayBuf.ar(1,bufnum,
		BufRateScale.kr(bufnum),
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(0.01, dur),doneAction:2)), 0.7);
	rev = FreeVerb.ar(grain, 0.33, 0.9, 0.5, 0.2);
	filter = Resonz.ar(rev, fil_freq);
	Out.ar(0, Pan2.ar(filter, pan))}).add;

SynthDef(\sfgranos, {
	|bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, atack = 0.5, rate=0.5|
	var grain, rev;
	grain= PlayBuf.ar(1,bufnum,
		//BufRateScale.kr(bufnum),
		rate.midiratio,
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(atack, dur),doneAction:2));
	rev = FreeVerb.ar(grain, 0.2, 0.5, 0.5, amp);
	Out.ar(0,Pan2.ar(rev, pan))}).add;

SynthDef(\sfgranos1, {
	|bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, atack = 0.5|
	var grain, rev;
	grain= PlayBuf.ar(1,bufnum,
		BufRateScale.kr(bufnum),
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(atack, dur),doneAction:2));
	rev = FreeVerb.ar(grain, 0.3, 0.9, 0.5, amp);
	Out.ar(0,Pan2.ar(rev, pan))}).add;

SynthDef(\sfgranos2, {
	|bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, atack = 0.5|
	var grain, rev;
	grain= PlayBuf.ar(1,bufnum,
		BufRateScale.kr(bufnum),
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(atack, dur),doneAction:2));
	rev = FreeVerb.ar(grain, 0.01, 0.5, 0.5, 0.1);
	Out.ar(0,Pan2.ar(rev, pan)*0.5)}).add;

SynthDef(\sfgranos3, {
	|bufnum=1, pan=0.0, startPos=0.0, amp=0.1, dur=0.04, atack = 0.5|
	var grain, rev;
	grain= PlayBuf.ar(1,bufnum,
		BufRateScale.kr(bufnum),
		1,
		BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(atack, dur),doneAction:2));
	rev = FreeVerb.ar(grain, 0.01, 0.5, 0.5, 0.1);
	Out.ar(0,Pan2.ar(rev, pan)*0.4)}).add;

SynthDef(\Klank, { arg out=0, i_freq, amps=0.5, gate = 1;
	var klank, n, harm, amp, ring, env;
	// harmonics
	harm = \harm.ir(Array.series(16, 1, 1).postln);
	// amplitudes
	amp = \amp.ir(Array.fill(16, 0.1));
	// ring times
	ring = \ring.ir(Array.fill(16, 1));
	klank = Klank.ar(`[harm, amp, ring], {PinkNoise.ar(0.001)}.dup, i_freq);
	env = EnvGen.kr(Env.new([0, 2, 15, 0], [5, 23, 5]), gate, doneAction: 2);
	Out.ar(out, env*klank);
}).add;