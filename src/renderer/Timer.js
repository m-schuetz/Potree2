
let capacity = 100;
let stamps = [];
let initialized = false;
let querySet = null;
let queryBuffer = null;
let index = 0;
let unresolvedIndex = 0;
let frame = 0;

let enabled = true;

let counters = {};

export function setEnabled(value){
	enabled = value;
}

function init(renderer){
	querySet = renderer.device.createQuerySet({
		type: "timestamp",
		count: capacity,
	});

	queryBuffer = renderer.device.createBuffer({
		size: 8 * capacity,
		usage: GPUBufferUsage.QUERY_RESOLVE 
			| GPUBufferUsage.STORAGE
			| GPUBufferUsage.COPY_SRC
			| GPUBufferUsage.COPY_DST,
	});

	initialized = true;
}

function frameStart(renderer){

	if(!enabled){
		return;
	}

	index = 0;
	unresolvedIndex = 0;

	if(!initialized){
		init(renderer);
	}

	timestampSep(renderer, "frame-start");

}

function frameEnd(renderer){

	if(!enabled){
		return;
	}

	timestampSep(renderer, "frame-end");

	let localStamps = stamps

	renderer.readBuffer(queryBuffer, 0, 8 * index).then( buffer => {

		// if((frame % 30) !== 0){
		// 	return;
		// }

		let u64 = new BigInt64Array(buffer);

		let tStart = u64[0];
		let starts = new Map();
		let durations = [];

		// let msg = "timestamps: \n";
		for(let i = 0; i < Math.min(u64.length, localStamps.length); i++){

			let label = localStamps[i].label;
			let current = Number(u64[i] - tStart) / 1_000_000;
			
			if(label.endsWith("-start")){
				starts.set(label, u64[i]);
			}else if(label.endsWith("-end")){
				let lblBase = label.replace("-end", "");
				let tStart = starts.get(lblBase + "-start");

				let current = Number(u64[i] - tStart) / 1_000_000;
				let message = `${lblBase}: ${current.toFixed(3)}ms`;
				durations.push(message);

				if(!counters[lblBase]){
					counters[lblBase] = [current];
				}else{
					counters[lblBase].push(current);
				}

				let doPrint = localStamps[i].args?.print ?? false;
				if(doPrint){
					console.log(message);
				}
			}
		}

		if((frame % 30) === 0){

			let msg = "durations: \n";

			for(let label of Object.keys(counters)){
				let values = counters[label];

				let min = Infinity;
				let max = 0;
				let sum = 0;

				for(let value of values){
					min = Math.min(min, value);
					max = Math.max(max, value);
					sum += value;
				}

				let avg = sum / values.length;

				msg += `[${label}]: avg: ${avg.toFixed(1)}, min: ${min.toFixed(1)}, max: ${max.toFixed(1)}\n`;

			}

			document.getElementById("msg_dbg").innerText = msg;

			counters = {};
		}


	});

	frame++;
	stamps = [];
}

function timestamp(encoder, label, args = {}){

	if(!enabled){
		return;
	}

	encoder.writeTimestamp(querySet, index);
	stamps.push({label, index, args});

	index++;
};

function timestampSep(renderer, label){

	if(!enabled){
		return;
	}

	let commandEncoder = renderer.device.createCommandEncoder();

	timestamp(commandEncoder, label);
	resolve(renderer, commandEncoder);

	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);
};

function resolve(renderer, commandEncoder){

	if(!enabled){
		return;
	}

	let first = unresolvedIndex;
	let count = index - unresolvedIndex;
	commandEncoder.resolveQuerySet(querySet, first, count, queryBuffer, 8 * first);

	unresolvedIndex = index;
}



export {
	frameStart,
	frameEnd,
	timestamp,
	timestampSep,
	resolve,
};


