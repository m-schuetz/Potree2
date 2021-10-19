

import {BrotliDecode} from "../../../libs/brotli/decode.js";


onmessage = async function (e) {

	let url = e.data.url;
	let {posOffset, posSize, colOffset, colSize} = e.data;

	try{
		let pPosition = fetch(url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${posOffset}-${posOffset + posSize - 1}`,
			},
		});
		let pColor = fetch(url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${colOffset}-${colOffset + colSize - 1}`,
			},
		});


		let loadSize = ((posSize + colSize) / (1024 ** 2)).toFixed(1);
		console.log(`loading ${loadSize} MB in 2 requests`);


		let [rPosition, rColor] = await Promise.all([pPosition, pColor]);
		let [bPosition, bColor] = await Promise.all([rPosition.arrayBuffer(), rColor.arrayBuffer()]);

		let position_decoded = BrotliDecode(new Int8Array(bPosition));
		let color_decoded = BrotliDecode(new Int8Array(bColor));
		let message = {position_decoded, color_decoded};
		let transferables = [position_decoded.buffer, color_decoded.buffer];

		postMessage(message, transferables);
	}catch(e){

	}

}