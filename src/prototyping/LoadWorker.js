

onmessage = async function(e) {
	
	let tStart = performance.now();

	let files = e.data;

	let printElapsed = (label) => {
		let duration = performance.now() - tStart;
		console.log(`${label}: ${(duration / 1000).toFixed(3)}s`);
	};

	let blobs = [];
	for (let file of files) {
		let promise = file.slice(0, 10);
		blobs.push(promise);

		if(blobs.length >= 32){
			break;
		}
	}

	printElapsed("00");
	await Promise.all(blobs);
	printElapsed("10");


	let buffers = blobs.map(blob => blob.arrayBuffer());
	printElapsed("20");
	await Promise.all(buffers);
	printElapsed("30");

	let duration = performance.now() - tStart;
	console.log(`duration: ${(duration / 1000).toFixed(3)}s`);

}