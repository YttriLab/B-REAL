import * as mp4boxModule from "./node_modules/mp4box/dist/mp4box.all.js";

// Handle both default export and named exports
const MP4Box = mp4boxModule.default || mp4boxModule;

// DataStream might be at different locations depending on mp4box version
const DataStream = MP4Box.DataStream || mp4boxModule.DataStream || globalThis.DataStream;

let mp4boxfile = null;
let videoTrack = null;
let nbSamples = 200;

const postError = (message) => {
  console.error("[mp4box-worker] Error:", message);
  self.postMessage({ type: "error", message });
};

self.onmessage = (event) => {
  const data = event.data || {};
  
  switch (data.type) {
    case "init": {
      console.log("[mp4box-worker] Initializing mp4box...");
      nbSamples = Number(data.nbSamples) || 200;
      videoTrack = null;
      
      try {
        mp4boxfile = MP4Box.createFile();
      } catch (err) {
        postError(`Failed to create MP4Box file: ${err.message || err}`);
        return;
      }
      
      if (!mp4boxfile) {
        postError("MP4 demuxer failed to initialize (createFile returned null).");
        return;
      }
      
      console.log("[mp4box-worker] mp4boxfile created successfully");
      
      mp4boxfile.onError = (error) => {
        postError(`MP4Box error: ${error && error.message ? error.message : error}`);
      };
      
      mp4boxfile.onMoovStart = () => {
        console.log("[mp4box-worker] moov atom parsing started");
        self.postMessage({ type: "moovStart" });
      };
      
      mp4boxfile.onReady = (info) => {
        console.log("[mp4box-worker] onReady called, tracks:", info.tracks.length);
        
        videoTrack = info.tracks.find((track) => track.type === "video");
        if (!videoTrack) {
          postError("No video track found.");
          return;
        }
        
        console.log("[mp4box-worker] Video track found:", videoTrack.id, videoTrack.codec);
        
        // Extract codec description (avcC/hvcC) for WebCodecs VideoDecoder
        let description = null;
        try {
          const trak = mp4boxfile.getTrackById(videoTrack.id);
          console.log("[mp4box-worker] Got track object:", !!trak);
          
          if (trak && trak.mdia && trak.mdia.minf && trak.mdia.minf.stbl && trak.mdia.minf.stbl.stsd) {
            const entries = trak.mdia.minf.stbl.stsd.entries || [];
            console.log("[mp4box-worker] Sample description entries:", entries.length);
            
            for (const entry of entries) {
              console.log("[mp4box-worker] Entry keys:", Object.keys(entry).join(", "));
              const codecBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
              
              if (codecBox) {
                const boxType = entry.avcC ? "avcC" : entry.hvcC ? "hvcC" : entry.vpcC ? "vpcC" : "av1C";
                console.log("[mp4box-worker] Found codec box:", boxType);
                console.log("[mp4box-worker] DataStream available:", !!DataStream);
                
                if (DataStream) {
                  try {
                    const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
                    codecBox.write(stream);
                    
                    if (stream.buffer.byteLength > 8) {
                      description = new Uint8Array(stream.buffer, 8);
                      console.log("[mp4box-worker] Extracted description via DataStream:", description.byteLength, "bytes");
                    }
                  } catch (dsErr) {
                    console.warn("[mp4box-worker] DataStream serialization failed:", dsErr);
                  }
                }
                
                if (!description && codecBox.data) {
                  if (codecBox.data instanceof ArrayBuffer) {
                    description = new Uint8Array(codecBox.data);
                  } else if (codecBox.data.buffer) {
                    description = new Uint8Array(codecBox.data.buffer, codecBox.data.byteOffset, codecBox.data.byteLength);
                  }
                  if (description) {
                    console.log("[mp4box-worker] Extracted description from box.data:", description.byteLength, "bytes");
                  }
                }
                
                if (!description && typeof mp4boxfile.getCodecs === "function") {
                  console.log("[mp4box-worker] Codecs string:", mp4boxfile.getCodecs());
                }
                
                break;
              }
            }
            
            if (!description) {
              console.warn("[mp4box-worker] No codec config box found in sample descriptions");
              entries.forEach((e, i) => console.log(`[mp4box-worker] Entry ${i} type:`, e.type, e.codingname));
            }
          } else {
            console.warn("[mp4box-worker] Could not navigate to stsd entries");
            console.log("[mp4box-worker] trak:", !!trak, "mdia:", !!trak?.mdia, "minf:", !!trak?.mdia?.minf);
          }
        } catch (err) {
          console.error("[mp4box-worker] Failed to extract codec description:", err);
        }
        
        // Start extraction immediately (standard mp4box.js flow)
        mp4boxfile.setExtractionOptions(videoTrack.id, null, { nbSamples });
        console.log("[mp4box-worker] setExtractionOptions done for track", videoTrack.id, "nbSamples:", nbSamples);
        mp4boxfile.start();
        console.log("[mp4box-worker] start() called, extractedTracks:", mp4boxfile.extractedTracks?.length);
        
        const trackWithDescription = { ...videoTrack, description };
        try {
          self.postMessage(
            { type: "ready", track: trackWithDescription },
            description ? [description.buffer] : []
          );
          console.log("[mp4box-worker] 'ready' message sent successfully");
        } catch (readyErr) {
          console.error("[mp4box-worker] Failed to send 'ready':", readyErr);
          postError("Failed to send ready: " + (readyErr.message || readyErr));
        }
      };
      
      mp4boxfile.onSamples = (_id, _user, samples) => {
        try {
          console.log("[mp4box-worker] onSamples called, count:", samples.length);
          
          const cleanSamples = [];
          for (let si = 0; si < samples.length; si++) {
            const sample = samples[si];
            let dataCopy = null;
            if (sample.data && sample.data.byteLength > 0) {
              dataCopy = new Uint8Array(sample.data.byteLength);
              dataCopy.set(
                sample.data instanceof Uint8Array
                  ? sample.data
                  : new Uint8Array(sample.data)
              );
            }
            cleanSamples.push({
              data: dataCopy,
              size: sample.size,
              duration: sample.duration,
              cts: sample.cts,
              dts: sample.dts,
              timescale: sample.timescale,
              is_sync: sample.is_sync,
              number: sample.number,
              track_id: sample.track_id,
              dataType: dataCopy ? "Uint8Array" : null
            });
          }
          
          self.postMessage({ type: "samples", samples: cleanSamples });
          console.log("[mp4box-worker] Sent", cleanSamples.length, "samples");
        } catch (err) {
          console.error("[mp4box-worker] onSamples error:", err);
          self.postMessage({ type: "error", message: "onSamples failed: " + (err.message || err) });
        }
      };
      
      self.postMessage({ type: "initialized" });
      break;
    }
    
    case "chunk": {
      if (!mp4boxfile) {
        console.warn("[mp4box-worker] Received chunk but mp4boxfile is null");
        return;
      }
      
      const buffer = data.buffer;
      if (!buffer) {
        console.warn("[mp4box-worker] Received empty chunk buffer");
        return;
      }
      
      const fileStart = data.fileStart || 0;
      const isLast = Boolean(data.last);
      
      console.log("[mp4box-worker] Appending chunk: offset=" + fileStart + ", size=" + buffer.byteLength + ", last=" + isLast);
      buffer.fileStart = fileStart;
      
      let nextOffset;
      try {
        nextOffset = mp4boxfile.appendBuffer(buffer);
      } catch (err) {
        postError(`appendBuffer failed: ${err.message || err}`);
        return;
      }
      
      self.postMessage({
        type: "chunkAck",
        id: data.id,
        nextOffset: typeof nextOffset === "number" ? nextOffset : null
      });
      
      if (isLast) {
        console.log("[mp4box-worker] Last chunk received, calling flush()");
        mp4boxfile.flush();
        self.postMessage({ type: "flushed", end: true });
      }
      break;
    }
    
    case "flush": {
      if (mp4boxfile) {
        console.log("[mp4box-worker] Manual flush requested");
        mp4boxfile.flush();
        self.postMessage({ type: "flushed", end: Boolean(data.end) });
      }
      break;
    }
    
    case "stop": {
      if (mp4boxfile) {
        console.log("[mp4box-worker] Stop requested");
        mp4boxfile.stop();
      }
      break;
    }
    
    default:
      console.warn("[mp4box-worker] Unknown message type:", data.type);
      break;
  }
};

console.log("[mp4box-worker] Worker loaded (v3)");
console.log("[mp4box-worker] MP4Box available:", !!MP4Box);
console.log("[mp4box-worker] createFile:", typeof MP4Box?.createFile);
console.log("[mp4box-worker] DataStream:", typeof DataStream);
