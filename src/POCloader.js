


async function loadHierarchy(url){
	let responseHierarchy = await fetch(urlHierarchy);
	let hierarchyBuffer = await responseHierarchy.arrayBuffer();
	let view = new DataView(hierarchyBuffer);
	let i64array = BigUint64Array(arrayBuffer);

	let numNodes = hierarchyBuffer.byteLenght / 32;
	let nodes = [];
	
	for(let i = 0; i < numNodes; i++){
		let node = {
			byteOffset: i64array[4 * i + 0],
			byteLength: i64array[4 * i + 1],
			childPosition: i64array[4 * i + 2],
			childMask: view.getUint8(32 * i + 24),
			name: "",
			children: [],
		};

		nodes.push(node);
	}

	nodes[0].name = "r";

	for(let i = 0; i < numNodes; i++){

	}
	
}


export class POCLoader{

	constructor(){

	}

	static async load(url){


		let responseMetadata = await fetch(url);
		let json = await responseMetadata.json();

		let urlHierarchy = `${url}/../hierarchy.bin`;
		await loadHierarchy(urlHierarchy);

		console.log(json);

	}

};