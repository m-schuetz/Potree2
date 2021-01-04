
export class PointCloudOctreeNode{
	constructor(name){
		this.name = name;
		this.loaded = false;
		this.children = new Array(8).fill(null);
		this.level = 0;
	}
}