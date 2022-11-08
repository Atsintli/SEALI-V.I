(
s.options.numBuffers = 1024 * 128;
o = Server.local.options;
//Server.local.options.device = "Jack";
Server.local.options.memSize = 2.pow(21);
Server.internal.options.memSize = 2.pow(21);
s.reboot;
)

(
~currentPath = thisProcess.nowExecutingPath.dirname;
(~currentPath++"/10_get_audios_&_syths_SEALI.scd").load;
)

~largos[(0..~largos.size).choose] //only for test

(
~seali = NetAddr("127.0.0.1"); // loopback ----
OSCdef (\osc_seali, {|msg, time, addr, recvPort|
	var allData, array, flatness, loudness, centroid, mfcc0,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,file;
	# allData, array, flatness, loudness, centroid,
	mfcc0,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,file = msg;
	file = file.asInteger.postln;
	"   ".postln;
	"flatness".postln;
	flatness.postln;
	"loudness".postln;
	loudness.postln;
	"centroid".postln;
	centroid.postln;
	"mfcc0".postln;
	mfcc0.postln;
	"mfcc1".postln;
	mfcc1.postln;
	"mfcc2".postln;
	mfcc2.postln;
	"mfcc3".postln;
	mfcc3.postln;
	"file".postln;
	file.postln;
	"   ".postln;

	if(((mfcc0>0.3)&&(mfcc1>0.3)&&(centroid>0.4)),

		{"Clasificación".postln

			(
				~flauta = NetAddr("127.0.0.1"); // loopback ----

				OSCdef (\osc_violin, {|msg, time, addr, recvPort|
					var clase, ataques, cress, largos, tremolando;
					# clase, ataques, cress, largos, tremolando = msg;

					if(ataques > 0.99){
						"Ataques".postln;
						(
							{
								~buffselect = ~ataques[0..~ataques.size].choose;

								50.do{arg i;
									var timeprop = (i/199.0)**3;
									Synth(\sfgrain2,[
										\bufnum, ~buffselect,
										\startPos,rrand(0.0,timeprop),
										\dur, exprand(0.04, 0.7),
										\atack, exprand(0.5, 1),
										\amp, exprand(0.05,0.7),
										\pan, 1.0.rand2]);
									rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
								};
							}.fork
						)
					};

					if ((ataques > 0.5) && (ataques < 0.9),{
						"ataques 2".postln;
						if (cress > 0.01)
						{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

						{
							~buffselect = ~ataques[0..~ataques.size].choose;

							50.do{arg i;
								var prop, timestart, timeend;
								prop= (i%200)/200;
								timestart= prop*0.8;
								timeend= prop*(0.8+(0.1*~lastval));
								Synth(\sfgrain,[
									\bufnum, ~buffselect,
									\attack, exprand(0.05,2),
									\startPos,rrand(timestart,timeend),
									\amp, exprand(0.05,0.5),
									\pan, ~lastval.rand2,
									\dur, 1+(~lastval*0.5)]);
								//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
								exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
							};
						}.fork;
					};
					);

					if((ataques > 0.051) && (ataques < 0.49),{
						"ataques 3".postln;
						Synth(\Klank, [\i_freq, (largos*500).rand])
					}
					);

					if((cress > 0.1) && (cress < 0.89),{
						"cress".postln;
						if (cress > 0.1)
						{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

						{
							~buffselect = ~cress[0..~cress.size].choose;

							40.do{arg i;
								var prop, timestart, timeend;
								prop= (i%200)/200;
								timestart= prop*0.8;
								timeend= prop*(0.8+(0.1*~lastval));
								Synth(\sfgrain,[
									\bufnum, ~buffselect,
									\attack, exprand(0.05,1),
									\startPos,rrand(timestart,timeend),
									\amp, exprand(0.005,0.05),
									\pan, ~lastval.rand2,
									\dur, 1+(~lastval*0.5)]);
								//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
								exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
							};
						}.fork;
					}
					);


					if((cress > 0.9) && (cress < 1.1),{
						"Sinegrain".postln
						(
							{
								50.do{|i|
									var timeprop = (i/100.0)*4;
									Synth(\sinegrain2,[
										\freq, exprand(cress*8000+(4*i),0.5*301+(8*i)),
										\amp, exprand(0.01,0.05),
										\atack, exprand(0.1, 0.9),
										\pan, exprand(0.8, 0.0001)]);
									rrand((timeprop*0.05).max(largos),timeprop*0.01).wait;
									//rrand((timeprop*0.01).max(0.01),timeprop*0.01).wait
								};
							}.fork;
						)
					}
					);

					if((largos > 0.32) && (largos < 1.1),{
						"largos".postln;
						Synth(\tremolando, [
							\buf, ~largos[0..~largos.size].choose,
							\pan, ataques,
							\rate, (-12..12).choose.midiratio])
					}
					);

					if((tremolando > 0.32) && (tremolando < 1.1),{
						"tremolando 2".postln;
						Synth(\largos, [
							\buf, ~cress[0..~cress.size].choose,
							\pan, tremolando,
							\rate, (-12..-12).choose.midiratio])
					}
					);

				},'/clase',recvPort: 5012);
			)

		},

		{
			"Predicción".postln;

			if((mfcc0 > 0.5)&&(mfcc0 < 1)){
				"\t SynthBuf".postln;
				Synth(\buf, [
					//\bufnum, ~dataBase[0..~dataBase.size].choose,
					\bufnum, ~dataBase[file],
					\rate, [-12,-7,-5,0].choose.midiratio,
					\a, [0],
					\s, [0.9],
					\d, [3.5],
					\r, [0],
					\times, [[0.5,1,0.1].choose, [4,2,3].choose, [0.5,0.1,0.01].choose],
				]);

				"\t \t + Large Grain".postln;

				{
					~file = ~dataBase[file];
					10.do{arg i;
						var timeprop = (i/390.0)**2;
						Synth(\sfgranos,[
							\bufnum, ~file,
							\startPos,rrand(0.0,timeprop),
							\dur, exprand(3, 5),
							\rate, [-24-12,-7,0,12].choose,
							\atack, exprand(2.1, 0.6),
							\amp, exprand(0.08, 0.4),
							\pan, [-0.7,-0.5,-0.3,0,0.30,0.5,0.7].choose]);
						[rrand((timeprop*100).max(0.5**mfcc1),timeprop*2), 0.1].choose.wait
					};
				}.fork
			};

			if((mfcc1 > 0.7)&&(loudness < 0.9)){
				"Large Grain".postln;
				(
					{
						~file = ~dataBase[file];
						10.do{arg i;
							var timeprop = (i/390.0)**2;
							Synth(\sfgranos,[
								\bufnum, ~file,
								\startPos,rrand(0.0,timeprop),
								\dur, exprand(3, 5),
								\rate, [-24-12,-7,0,12].choose,
								\atack, exprand(0.1, 0.5),
								\amp, 0.07,
								\pan, [-0.7,-0.5,-0.3,0,0.30,0.5,0.7].choose]);
							[rrand((timeprop*100).max(0.1**mfcc1),timeprop*2), 0.1].choose.wait
						};
					}.fork
				)
			};

			if(mfcc3 == 0.10000000149012){
				"Large Grain 2".postln;
				(
					{
						~file = ~dataBase[file];
						10.do{arg i;
							var timeprop = (i/(500))**2;
							Synth(\sfgranos1,[
								\bufnum, ~file,
								\startPos,rrand(0.0,timeprop),
								\dur, exprand(1, 2),
								\rate, [-24-12,0-12,24].choose,
								\atack, exprand(2, 1),
								\amp, if(loudness > 0.4,{exprand(0.07, 0.1)}, {exprand(0.1,0.5)}),
								\pan, [-0.99,-0.7,-0.5,-0.3,0,0.30,0.5,0.7,0.99].choose]);
							[rrand((timeprop*0.1).max(0.1**0.6),timeprop*1), 0.01].choose.wait
						};
					}.fork
				)
			};

			if((centroid > 0.4)&&(centroid < 0.6)){
				"Texture Grain".postln;

				(
					{
						~file = ~dataBase[file];
						20.do{arg i;
							var timeprop = (i/200.0)**2;
							Synth(\sfgranos2,[
								\bufnum, ~file,
								\startPos, 0.01,
								\dur, exprand(8,4),
								\atack, 0.01,
								\amp, exprand(0.1,0.5),
								\pan, [-0.6,-0.2,0.1,0,0.1,0.2,0.5,0.6].choose]);
							rrand((timeprop*160).max(1**mfcc1),timeprop*1).wait
						};
					}.fork
				)
			};

			if(mfcc3 > 0.6){
				"Small Grain".postln;

				(
					{
						~file = ~dataBase[file];
						10.do{arg i;
							var timeprop = (i/399.0)**2;
							Synth(\sfgranos1,[
								\bufnum, ~file,
								\startPos,rrand(0.0,timeprop),
								\dur, exprand(0.001, 0.05),
								\atack,exprand(0.1, 0.5),
								\amp, exprand(0.1,0.5),
								\pan, [-0.9,-0.6,-0.2,0.1,0,0.1,0.2,0.5,0.6,0.9].choose]);
							rrand((timeprop*50).max(0.1**mfcc1),timeprop*2).wait
						};
					}.fork
				)
			};


			if((mfcc3 > 0.4)&&(mfcc3 < 0.6)){
				"Drone Grain".postln;

				(
					{
						~file = ~dataBase[file];
						10.do{arg i;
							var timeprop = (i/399.0)**2;
							Synth(\sfgranos3,[
								\bufnum, ~file,
								\startPos, 0.00,
								\dur, exprand(0.1, 1),
								\atack, exprand(1, 1.9),
								\amp, if(loudness > 0.2,{exprand(0.02, 0.07)}, {exprand(0.1,0.6)}),
								\pan, [-0.7,-0.2,0.1,0,0.1,0.2,0.5,0.7].choose]);
							[rrand((timeprop*100).max(0.1**mfcc1),timeprop*2), 0.1].choose.wait
						};
					}.fork
				)
			};

			if((mfcc2 > 0.6) && (mfcc2 < 1),{
				"Klank".postln;
				Synth(\Klank, [
					\i_freq, (1000..2000).rand,
					\amps, (0.7).rand
				])
			}
			);

		},

	);

},'/allData',recvPort: 5008);
)


//otras posibilidades de sonificación//

(
~seali = NetAddr("127.0.0.1"); // loopback ----
OSCdef (\osc_seali, {|msg, time, addr, recvPort|
	var allData, array, flatness, loudness, centroid, mfcc0,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,file;
	# allData, array, flatness, loudness, centroid,
	mfcc0,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,file = msg;
	file = file.asInteger.postln;
	"   ".postln;
	"flatness".postln;
	flatness.postln;
	"loudness".postln;
	loudness.postln;
	"centroid".postln;
	centroid.postln;
	"mfcc0".postln;
	mfcc0.postln;
	"mfcc1".postln;
	mfcc1.postln;
	"mfcc2".postln;
	mfcc2.postln;
	"mfcc3".postln;
	mfcc3.postln;
	"file".postln;
	file.postln;
	"   ".postln;

	if(((mfcc0>0.3)&&(mfcc1>0.4)&&(flatness>=0.2)),{"Menor actividad".postln;

		OSCdef (\osc_violin).clear;

		if((mfcc0 > 0.5)&&(mfcc0 < 1)){
			"SynthBuf".postln;
			Synth(\buf, [
				//\bufnum, ~dataBase[0..~dataBase.size].choose,
				\bufnum, ~dataBase[file],
				\rate, [-12,-7,-5,0].choose.midiratio,
				\a, [0],
				\s, [0.5],
				\d, [1.5],
				\r, [0],
				//\times, [[0.5,1,0.1].choose, [4,2,3].choose, [0.5,0.1,0.01].choose],
				\times, [1,2,0.5],
			]);

			"\t + Large Grain".postln;

			{
				~file = ~dataBase[file];
				10.do{arg i;
					var timeprop = (i/390.0)**2;
					Synth(\sfgranos,[
						\bufnum, ~file,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(3, 5),
						\rate, [-24-12,-7,0,12].choose,
						\atack, exprand(2.1, 0.6),
						\amp, exprand(0.08, 0.4),
						\pan, [-0.7,-0.5,-0.3,0,0.30,0.5,0.7].choose]);
					[rrand((timeprop*100).max(0.5**mfcc1),timeprop*2), 0.1].choose.wait
				};
			}.fork
		};

		if((mfcc1 > 0.7)&&(loudness < 0.9)){
			"Large Grain".postln;
			(
				{
					~file = ~dataBase[file];
					10.do{arg i;
						var timeprop = (i/390.0)**2;
						Synth(\sfgranos,[
							\bufnum, ~file,
							\startPos,rrand(0.0,timeprop),
							\dur, exprand(3, 5),
							\rate, [-24-12,-7,0,12].choose,
							\atack, exprand(0.1, 0.5),
							\amp, 0.07,
							\pan, [-0.7,-0.5,-0.3,0,0.30,0.5,0.7].choose]);
						[rrand((timeprop*100).max(0.1**mfcc1),timeprop*2), 0.1].choose.wait
					};
				}.fork
			)
		};

		if(mfcc3 == 0.10000000149012){
			"Large Grain 2".postln;
			(
				{
					~file = ~dataBase[file];
					10.do{arg i;
						var timeprop = (i/(500))**2;
						Synth(\sfgranos1,[
							\bufnum, ~file,
							\startPos,rrand(0.0,timeprop),
							\dur, exprand(1, 2),
							\rate, [-24-12,0-12,24].choose,
							\atack, exprand(2, 1),
							\amp, if(loudness > 0.4,{exprand(0.07, 0.1)}, {exprand(0.1,0.5)}),
							\pan, [-0.99,-0.7,-0.5,-0.3,0,0.30,0.5,0.7,0.99].choose]);
						[rrand((timeprop*0.1).max(0.1**0.6),timeprop*1), 0.01].choose.wait
					};
				}.fork
			)
		};

		if((centroid > 0.4)&&(centroid < 0.6)){
			"Texture Grain".postln;

			(
				{
					~file = ~dataBase[file];
					10.do{arg i;
						var timeprop = (i/200.0)**2;
						Synth(\sfgranos2,[
							\bufnum, ~file,
							\startPos, 0.01,
							\dur, exprand(8,4),
							\atack, 0.01,
							\amp, exprand(0.1,0.5),
							\pan, [-0.6,-0.2,0.1,0,0.1,0.2,0.5,0.6].choose]);
						rrand((timeprop*160).max(1**mfcc1),timeprop*1).wait
					};
				}.fork
			)
		};

		if(mfcc3 > 0.6){
			"Small Grain".postln;

			(
				{
					~file = ~dataBase[file];
					10.do{arg i;
						var timeprop = (i/399.0)**2;
						Synth(\sfgranos1,[
							\bufnum, ~file,
							\startPos,rrand(0.0,timeprop),
							\dur, exprand(0.001, 0.05),
							\atack,exprand(0.1, 0.5),
							\amp, exprand(0.1,0.5),
							\pan, [-0.9,-0.6,-0.2,0.1,0,0.1,0.2,0.5,0.6,0.9].choose]);
						rrand((timeprop*50).max(0.1**mfcc1),timeprop*2).wait
					};
				}.fork
			)
		};


		if((mfcc3 > 0.4)&&(mfcc3 < 0.6)){
			"Drone Grain".postln;

			(
				{
					~file = ~dataBase[file];
					50.do{arg i;
						var timeprop = (i/399.0)**2;
						Synth(\sfgranos3,[
							\bufnum, ~file,
							\startPos, 0.00,
							\dur, exprand(0.1, 1),
							\atack, exprand(1, 1.9),
							\amp, if(loudness > 0.2,{exprand(0.02, 0.07)}, {exprand(0.1,0.6)}),
							\pan, [-0.7,-0.2,0.1,0,0.1,0.2,0.5,0.7].choose]);
						[rrand((timeprop*100).max(0.1**mfcc1),timeprop*2), 0.1].choose.wait
					};
				}.fork
			)
		};

		if((mfcc2 > 0.6) && (mfcc2 < 1),{
			"- \Klank".postln;
			Synth(\Klank, [
				\i_freq, (1000).rand,
				\amps, (0.3).rand
			])
		}
		);


	},

	{"Mayor actividad".postln

		(
			~violin = NetAddr("127.0.0.1"); // loopback ----

			OSCdef (\osc_violin, {|msg, time, addr, recvPort|
				var clase, ataques, cress, largos, tremolando;
				# clase, ataques, cress, largos, tremolando = msg;

				if(ataques > 0.99){
					"CLASE 0 Fluata".postln;
					(
						{
							~buffselect = ~ataques[0..~ataques.size].choose;

							50.do{arg i;
								var timeprop = (i/199.0)**3;
								Synth(\sfgrain2,[
									\bufnum, ~buffselect,
									\startPos,rrand(0.0,timeprop),
									\dur, exprand(0.04, 0.7),
									\atack, exprand(0.5, 1),
									\amp, exprand(0.05,0.7),
									\pan, 1.0.rand2]);
								rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
							};
						}.fork
					)
				};

				if ((ataques > 0.5) && (ataques < 0.9),{
					"Largos".postln;
					if (cress > 0.01)
					{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

					{
						~buffselect = ~largos[0..~largos.size].choose;

						50.do{arg i;
							var prop, timestart, timeend;
							prop= (i%200)/200;
							timestart= prop*0.8;
							timeend= prop*(0.8+(0.1*~lastval));
							Synth(\sfgrain,[
								\bufnum, ~buffselect,
								\attack, exprand(0.05,2),
								\startPos,rrand(timestart,timeend),
								\amp, exprand(0.05,0.5),
								\pan, ~lastval.rand2,
								\dur, 1+(~lastval*0.5)]);
							//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
							exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
						};
					}.fork;
				};
				);

				if((ataques > 0.051) && (ataques < 0.49),{"klank".postln;
					Synth(\Klank, [\i_freq, (largos*500).rand])
				}
				);

				if((cress > 0.1) && (cress < 0.89),{
					"Electronic".postln;
					if (cress > 0.1)
					{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

					{
						~buffselect = ~dataBase[0..~dataBase.size].choose;

						40.do{arg i;
							var prop, timestart, timeend;
							prop= (i%200)/200;
							timestart= prop*0.8;
							timeend= prop*(0.8+(0.1*~lastval));
							Synth(\sfgrain,[
								\bufnum, ~buffselect,
								\attack, exprand(0.05,1),
								\startPos,rrand(timestart,timeend),
								\amp, exprand(0.005,0.05),
								\pan, ~lastval.rand2,
								\dur, 1+(~lastval*0.5)]);
							//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
							exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
						};
					}.fork;
				}
				);

				if((cress > 0.999) && (cress < 1.1),{
					"Sinegrain2".postln
					(
						{
							50.do{|i|
								var timeprop = (i/100.0)*4;
								Synth(\sinegrain2,[
									\freq, exprand(3500+(4*i),0.5*301+(8*i)),
									\amp, exprand(0.1,0.3),
									\atack, exprand(0.1, 0.9),
									\pan, exprand(0.8, 0.0001)]);
								rrand((timeprop*0.05).max(largos),timeprop*0.01).wait;
								//rrand((timeprop*0.01).max(0.01),timeprop*0.01).wait
							};
						}.fork;
					)
				}
				);

				if((largos > 0.32) && (largos < 1.1),{
					"Tremolando".postln;
					Synth(\tremolando, [
						\buf, ~tremolando[0..~tremolando.size].choose,
						\pan, ataques,
						\rate, (-12..12).choose.midiratio])
				}
				);

			},'/clase',recvPort: 5012);
		)

	});

},'/allData',recvPort: 5008);
);

