
import {Potree, Mesh, Vector3, Vector4, geometries, SceneNode} from "potree";
import {EventDispatcher, KeyCodes, MouseCodes} from "potree";

let counter = 0;

export class Measure{

	constructor(){
		this.label = `Measure ${counter}`;
		this.markers = [];
		this.markers_highlighted = [];
		this.requiredMarkers = 1;
		this.maxMarkers = 1;
		this.showEdges = true;
		// this.showEdgesClosed = false;

		counter++;
	}

	addMarker(position){
		this.markers.push(position.clone());
	}

};

export class PointMeasure extends Measure{

	constructor(){
		super();
	}

	addMarker(position){
		this.markers.push(position.clone());
	}
};

export class DistanceMeasure extends Measure{

	constructor(){
		super();
		this.requiredMarkers = 0;
		this.maxMarkers = 100;
	}

	addMarker(position){
		this.markers.push(position.clone());
	}

};

export class HeightMeasure extends Measure{

	constructor(){
		super();
		this.requiredMarkers = 2;
		this.maxMarkers = 2;
	}

	addMarker(position){
		this.markers.push(position.clone());

	}

};

export class MeasureTool{

	constructor(potree){
		this.potree = potree;
		this.renderer = potree.renderer;
		this.element = potree.renderer.canvas;

		this.node = new SceneNode("MeasureTool");
		this.cursor = new Mesh("MeasureTool_cursor", geometries.sphere);
		this.cursor.scale.set(10.5, 10.5, 10.5);
		this.cursor.position.set(67.97, -4.54, -23.56);
		// this.cursor.renderLayer = 10;
		this.cursor.visible = true;

		this.node.children.push(this.cursor);
		potree.scene.root.children.push(this.node);

		this.currentMeasurement = null;

		this.measures = [];

		potree.onUpdate(this.update.bind(this));

		this.dispatcher = new EventDispatcher();

	}

	reset(){

	}

	update(){
		let depth = camera.getWorldPosition().distanceTo(this.cursor.position);
		let radiusFactor = 50;
		let radius = depth / radiusFactor;

		if(this.currentMeasurement){
			// this.cursor.visible = true;
			// this.cursor.scale.set(radius, radius, radius);


			let args = {
				color: new Vector4(0, 1, 0, 1)
			};
			this.renderer.drawSphere(this.cursor.position, radius, args);
		}else{
			this.cursor.visible = false;
		}

		for(let measure of this.measures){

			// DRAW MARKERS
			for(let markerIndex = 0; markerIndex < measure.markers.length; markerIndex++){
				let marker = measure.markers[markerIndex];

				let depth = camera.getWorldPosition().distanceTo(marker);
				let radius = depth / radiusFactor;

				let args = {
					color: new Vector4(0, 1, 0, 1)
				};
				if(measure.markers_highlighted[markerIndex]){
					args.color.set(255, 127, 80, 255).multiplyScalar(1 / 255);
				}
				this.renderer.drawSphere(marker, radius, args);
			}

			// DRAW EDGES
			if(measure.showEdges){
				for(let i = 0; i < measure.markers.length - 1; i++){
					this.renderer.drawLine(
						measure.markers[i + 0],
						measure.markers[i + 1],
						new Vector3(255, 0, 0),
					);
				}
			}

			// DRAW HEIGHT MEASURE
			if(measure instanceof HeightMeasure && measure.markers.length === 2){

				let low  = measure.markers[0];
				let high = measure.markers[1];
				if(low.z > high.z){
					[low, high] = [high, low];
				}

				let start = new Vector3(high.x, high.y, high.z);
				let end = new Vector3(high.x, high.y, low.z);

				this.renderer.drawLine(start, end, new Vector3(0, 0, 255));

				this.renderer.drawLine(low, end, new Vector3(255, 0, 0));
			}

		}
	}

	measureMove(e){
		let {x, y} = e.mouse;

		let node = this.cursor;

		Potree.pick(x, y, (result) => {

			if(result.depth !== Infinity){
				node.position.copy(result.position);
				node.visible = true;

				Potree.pickPos = result.position;
			}
		});
	}

	onClick(e){
		console.log(e);
	}

	startMeasuring(measure){

		if(this.currentMeasurement){
			this.stopMeasuring();
		}

		if(!measure){
			measure = new Measure();
		}

		this.currentMeasurement = measure;

		this.measures.push(measure);

		this.dispatcher.add("mousemove", (e) => {this.measureMove(e)});
		this.dispatcher.add("mouseup", (e) => {
			
			if(e.event.button === MouseCodes.LEFT && this.cursor.visible){

				let markerPos = this.cursor.position.clone();
			
				measure.addMarker(markerPos);

				if(measure.markers.length === measure.maxMarkers){
					this.stopMeasuring();
				}else if(measure.markers.length === measure.requiredMarkers){
					this.stopMeasuring();
				}

			}else if(e.event.button === MouseCodes.RIGHT){

				if(measure.requiredMarkers && measure.markers.length !== measure.requiredMarkers){
					this.measures.pop();
				}

				this.stopMeasuring();
			}

		});

	}

	stopMeasuring(){
		this.dispatcher.removeAll();
		this.cursor.visible = false;
		this.currentMeasurement = null;
	}

};
