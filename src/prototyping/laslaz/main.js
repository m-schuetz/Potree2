
import {LASFile} from "./plasio/js/laslaz.js";


async function run(){

	//let response = await fetch("./bunny_small.laz");
	let response = await fetch("./ot_35120B4310A_1.laz");
	let buffer = await response.arrayBuffer();

	
	let lf = new LASFile(buffer);
	await lf.open();
	lf.isOpen = true;

	let header = await lf.getHeader();

	console.log(header);
	console.log(header.pointsCount.toLocaleString());

	let hasMoreData = true;
	while(hasMoreData){
		let data = await lf.readData(1_000_000, 0, 1);


		console.log("===");
		console.log(data.count.toLocaleString());
		console.log(data.buffer);


		hasMoreData = data.hasMoreData;
	}

}

run();