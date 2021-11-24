
export class Attribute_Misc{

	constructor(stats){
		this.stats = stats;

	}

};

export class Attribute_RGB{

	constructor(stats){
		this.stats = stats;
	}

};

export class Attribute_Scalar{

	constructor(stats){
		this.stats = stats;
		this.range = [stats.min, stats.max];
		this.filterRange = [-Infinity, Infinity];
	}

};

export class Attribute_Listing{

	constructor(stats){
		this.stats = stats;

	}

};

export class PointCloudMaterial{

	constructor(){

		this.attributes = new Map();

	}

	init(pointcloud){

		let statsList = pointcloud.root.geometry.statsList;

		for(let stats of statsList){

			let attribute = null;

			if(stats.name === "rgba"){
				attribute = new Attribute_RGB(stats);
			}else if(stats.name === "intensity"){
				attribute = new Attribute_Scalar(stats);
			}else if(stats.name === "point source id"){
				attribute = new Attribute_Scalar(stats);
			}else if(stats.name === "gps-time"){
				attribute = new Attribute_Scalar(stats);
			}else if(stats.name === "classification"){
				attribute = new Attribute_Listing(stats);
			}else{
				
			}

			this.attributes.set(stats.name, attribute);
		}

		{ // elevation

			let xyz = statsList.find(stats => ["XYZ", "position"].includes(stats.name));

			if(xyz){
				let stats = {
					name: "elevation",
					min: xyz.min[2],
					max: xyz.max[2],
				};
				let attribute = new Attribute_Scalar(stats);
				this.attributes.set(stats.name, attribute);
			}
		}

		console.log(statsList);

		console.log(this.attributes);
	}

};