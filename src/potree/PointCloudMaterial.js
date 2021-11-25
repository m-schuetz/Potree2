

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

const ListingSchemes = {
	LAS_CLASSIFICATION: {
		0:       { visible: true, name: 'never classified'  , color: [0.5,  0.5,  0.5,  1.0] },
		1:       { visible: true, name: 'unclassified'      , color: [0.5,  0.5,  0.5,  1.0] },
		2:       { visible: true, name: 'ground'            , color: [0.63, 0.32, 0.18, 1.0] },
		3:       { visible: true, name: 'low vegetation'    , color: [0.0,  1.0,  0.0,  1.0] },
		4:       { visible: true, name: 'medium vegetation' , color: [0.0,  0.8,  0.0,  1.0] },
		5:       { visible: true, name: 'high vegetation'   , color: [0.0,  0.6,  0.0,  1.0] },
		6:       { visible: true, name: 'building'          , color: [1.0,  0.66, 0.0,  1.0] },
		7:       { visible: true, name: 'low point(noise)'  , color: [1.0,  0.0,  1.0,  1.0] },
		8:       { visible: true, name: 'key-point'         , color: [1.0,  0.0,  0.0,  1.0] },
		9:       { visible: true, name: 'water'             , color: [0.0,  0.0,  1.0,  1.0] },
		12:      { visible: true, name: 'overlap'           , color: [1.0,  1.0,  0.0,  1.0] },
		DEFAULT: { visible: true, name: 'default'           , color: [0.3,  0.6,  0.6,  0.5] },
	},
	LAS_RETURN_NUMBER: {
		0:       { visible: true, name: '0'    , color: [0.5,  0.5,  0.5,  1.0] },
		1:       { visible: true, name: '1'    , color: [0.5,  0.5,  0.5,  1.0] },
		2:       { visible: true, name: '2'    , color: [0.63, 0.32, 0.18, 1.0] },
		3:       { visible: true, name: '3'    , color: [0.0,  1.0,  0.0,  1.0] },
		4:       { visible: true, name: '4'    , color: [0.0,  1.0,  0.0,  1.0] },
		5:       { visible: true, name: '5'    , color: [0.0,  1.0,  0.0,  1.0] },
		6:       { visible: true, name: '6'    , color: [0.0,  1.0,  0.0,  1.0] },
		7:       { visible: true, name: '7'    , color: [0.0,  1.0,  0.0,  1.0] },
		DEFAULT: { visible: true, name: 'default'           , color: [0.3,  0.6,  0.6,  0.5] },
	},
	NONE: {
		DEFAULT: { visible: true, name: 'default'           , color: [0.3,  0.6,  0.6,  0.5] },
	}
};

export class Attribute_Listing{

	constructor(stats){
		this.stats = stats;
		this.listing = ListingSchemes.NONE;
	}

	set(scheme){
		this.listing = scheme;
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
				attribute.set(ListingSchemes.LAS_CLASSIFICATION);
			}else if(stats.name === "return number"){
				attribute = new Attribute_Listing(stats);
				attribute.set(ListingSchemes.LAS_RETURN_NUMBER);
			}else if(stats.name === "scan angle"){
				attribute = new Attribute_Scalar(stats);
			}else{
				attribute = new Attribute_Scalar(stats);
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