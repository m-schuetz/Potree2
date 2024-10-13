
import {Potree, Mesh, Vector3, geometries, SceneNode} from "potree";
import {EventDispatcher, KeyCodes, MouseCodes} from "potree";

let counter = 0;

export class Measure{

	constructor(){
		this.label = `Measure ${counter}`;
		this.markers = [];
		this.requiredMarkers = 1;
		this.maxMarkers = 1;
		this.showEdges = true;
		// this.showEdgesClosed = false;

		counter++;
	}

	addMarker(position){
		this.markers.push(position.clone());
	}

	toHtml(){
		let htmlMarkers = "";
		for(let i = 0; i < this.markers.length; i++){

			let marker = this.markers[i];

			htmlMarkers += `
			<tr>
				<td style="text-align: right">${marker.x.toFixed(3)}</td>
				<td style="text-align: right">${marker.y.toFixed(3)}</td>
				<td style="text-align: right">${marker.z.toFixed(3)}</td>
			</tr>
			`;

		}

		let html = `
		<table style="width: 100%">
			${htmlMarkers}
		</table>
		`;
		
		return html;
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
		let radius = depth / 50;

		if(this.currentMeasurement){
			this.cursor.visible = true;
			this.cursor.scale.set(radius, radius, radius);
		}else{
			this.cursor.visible = false;
		}

		for(let measure of this.measures){

			for(let marker of measure.markers){

				let depth = camera.getWorldPosition().distanceTo(marker);
				let radius = depth / 50;

				this.renderer.drawSphere(marker, radius);
			}

			if(measure.showEdges){
				for(let i = 0; i < measure.markers.length - 1; i++){
					this.renderer.drawLine(
						measure.markers[i + 0],
						measure.markers[i + 1],
						new Vector3(255, 0, 0),
					);
				}
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
