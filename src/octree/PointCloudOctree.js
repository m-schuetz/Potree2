
export class Node{

	constructor(){
		this.buffers = null;
		this.boundingBox = null;
		this.children = [
			null, null, null, null,
			null, null, null, null,
		];
	}

}


export class PointCloudOctree{

	constructor(){
		this.name = "";
		this.loader = null;
		this.root = null;
		this.boundingBox = null;
	}

}