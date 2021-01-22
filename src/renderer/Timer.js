
let capacity = 100;
let stamps = [];
let initialized = false;
let querySet = null;
let queryBuffer = null;
let index = 0;
let unresolvedIndex = 0;
let frame = 0;

export let enabled = true;

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
	stamps = [];
	index = 0;
	unresolvedIndex = 0;

	if(!initialized){
		init(renderer);
	}

	timestampSep(renderer, "frame-start");

}

function frameEnd(renderer){

	timestampSep(renderer, "frame-end");

	renderer.readBuffer(queryBuffer, 0, 8 * index).then( buffer => {

		if((frame % 30) !== 0){
			return;
		}

		let u64 = new BigInt64Array(buffer);

		let tStart = u64[0];
		let starts = new Map();
		let durations = [];

		let msg = "timestamps: \n";
		for(let i = 0; i < Math.min(u64.length, stamps.length); i++){

			let label = stamps[i].label;
			let current = Number(u64[i] - tStart) / 1_000_000;
			
			if(label.endsWith("-start")){
				starts.set(label, u64[i]);
			}else if(label.endsWith("-end")){
				let lblBase = label.replace("-end", "");
				let tStart = starts.get(lblBase + "-start");

				let current = Number(u64[i] - tStart) / 1_000_000;
				durations.push(`${lblBase}: ${current.toFixed(3)}ms`);
			}

			msg += `${label}: ${current.toFixed(3)}ms\n`;
		}

		msg += "\ndurations: \n";
		msg += durations.join("\n");


		document.getElementById("msg_dbg").innerText = msg;
	});

	frame++;
}

function timestamp(encoder, label){
	encoder.writeTimestamp(querySet, index);
	stamps.push({label, index});

	index++;
};

function timestampSep(renderer, label){

	let commandEncoder = renderer.device.createCommandEncoder();

	timestamp(commandEncoder, label);
	resolve(renderer, commandEncoder);

	let commandBuffer = commandEncoder.finish();
	renderer.device.defaultQueue.submit([commandBuffer]);
};

function resolve(renderer, commandEncoder){
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


