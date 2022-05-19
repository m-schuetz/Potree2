import {EventDispatcher, SplatType} from "potree";

export class Attribute_Custom{
	constructor(name){
		this.name = name;
		this.stats = null;
		this.runtime = true;
		this.mapping = null;
	}
};

export class Attribute_Misc{
	constructor(stats){
		this.stats = stats;
		this.runtime = false;
		this.mapping = null;
	}
};

export class Attribute_RGB{

	constructor(stats){
		this.stats = stats;
		this.runtime = false;
		this.mapping = null;
	}

};

export class Attribute_Scalar{

	constructor(stats){
		this.stats = stats;
		// this.range = [stats.min, stats.max];
		// this.range = [0, 1];
		this.range = null;
		this.filterRange = [-Infinity, Infinity];
		this.clamp = false;
		this.runtime = false;
		this.mapping = null;
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
		this.mapping = null;
	}

	set(scheme){
		this.listing = scheme;
	}
};

export class PointCloudMaterial{

	constructor(){
		
		this.initialized = false;
		this.statsUpdated = false;
		this.attributes = new Map();
		this.mappings = new Map();
		this.needsCompilation = false;
		this.splatType = SplatType.POINTS;
		
		let dispatcher = new EventDispatcher();
		this.events = {
			dispatcher,
			onChange: (callback, args) => dispatcher.add("change", callback, args),
		};
	}

	recompile(){
		this.needsCompilation = true;
	}

	registerAttribute(name){

		if(this.attributes.has(name)){
			throw `an attribute with the id '${name}' is already registered`;
		}

		let attribute = new Attribute_Custom(name);

		this.attributes.set(name, attribute);

		this.recompile();
		this.events.dispatcher.dispatch("change", {material: this});
	}

	registerMapping({name, condition, wgsl}){

		if(this.mappings.has(name)){
			throw `a mapping with the id '${name}' is already registered`;
		}

		let mapping = {name, condition, wgsl};
		let index = 128 + this.mappings.size;

		this.mappings.set(name, mapping);

		this.recompile();
		this.events.dispatcher.dispatch("change", {material: this});

		for(let [name, attribute] of this.attributes){
			if(attribute.attribute && condition(attribute.attribute)){
				attribute.mapping = index;
			}
		}

		return index;
	}

	update(pointcloud){

		let statsList = pointcloud?.root?.geometry?.statsList;

		if(statsList && !this.statsUpdated){

			for(let stats of statsList){

				if(this.attributes.has(stats.name)){
					let attribute = this.attributes.get(stats.name);

					attribute.stats = stats;

					if(attribute instanceof Attribute_Scalar){
						attribute.range = [stats.min, stats.max];
					}
				}
			}

			{ // elevation

				let xyz = statsList.find(stats => ["XYZ", "position"].includes(stats.name));

				if(xyz){
					let stats = {
						name: "elevation",
						min: xyz.min[2],
						max: xyz.max[2],
					};
					let size = stats.max - stats.min;

					let attribute = this.attributes.get("elevation");
					attribute.stats = stats;
					attribute.clamp = true;
					attribute.range = [stats.min - 0.05 * size, stats.max + 0.05 * size];
				}
			}

			this.statsUpdated = true;

			this.events.dispatcher.dispatch("change", {material: this});
		}

		

	}

	init(pointcloud){

		if(this.initialized){
			return;
		}

		for(let attribute of pointcloud.attributes.attributes){

			let stats = null;

			let mapping = null;
			if(attribute.name === "rgba"){
				mapping = new Attribute_RGB(stats);
			}else if(attribute.name === "intensity"){
				mapping = new Attribute_Scalar(stats);
			}else if(attribute.name === "point source id"){
				mapping = new Attribute_Scalar(stats);
			}else if(attribute.name === "gps-time"){
				mapping = new Attribute_Scalar(stats);
			}else if(attribute.name === "classification"){
				mapping = new Attribute_Listing(stats);
				mapping.set(ListingSchemes.LAS_CLASSIFICATION);
			}else if(attribute.name === "return number"){
				mapping = new Attribute_Listing(stats);
				mapping.set(ListingSchemes.LAS_RETURN_NUMBER);
			}else if(attribute.name === "scan angle"){
				mapping = new Attribute_Scalar(stats);
			}else if(attribute.name === "Normal"){
				mapping = new Attribute_RGB(stats);
			}else{
				mapping = new Attribute_Scalar(stats);
			}

			mapping.attribute = attribute;

			this.attributes.set(attribute.name, mapping);
		}

		{ // elevation
			let stats = null;
			let attribute = new Attribute_Scalar(stats);
			attribute.clamp = true;
			attribute.runtime = true;
			this.attributes.set("elevation", attribute);
		}

		// console.log(statsList);
		console.log(this.attributes);

		this.initialized = true;

		this.events.dispatcher.dispatch("change", {material: this});
	}

};