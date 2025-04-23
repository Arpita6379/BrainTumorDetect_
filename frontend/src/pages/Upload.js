// import React, { useState } from 'react';
// import { useNavigate } from 'react-router-dom';
// import { uploadImage, getCurrentUser } from '../services/api';

// const Upload = () => {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [previewUrl, setPreviewUrl] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [results, setResults] = useState(null);
//   const [highlightedImageUrl, setHighlightedImageUrl] = useState(null);
//   const navigate = useNavigate();
//   const currentUser = getCurrentUser();

//   // Redirect if not logged in
//   if (!currentUser) {
//     navigate('/login');
//     return null;
//   }

//   const handleFileChange = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       setSelectedFile(file);
//       setPreviewUrl(URL.createObjectURL(file));
//       setError('');
//       setResults(null);
//       setHighlightedImageUrl(null);
//     }
//   };

//   const handleUpload = async (e) => {
//     e.preventDefault();
    
//     if (!selectedFile) {
//       setError('Please select an image first');
//       return;
//     }

//     setLoading(true);
//     setError('');

//     try {
//       const response = await uploadImage(selectedFile, currentUser.id);
      
//       // Store the ML results
//       setResults(response.ml_results);
      
//       // If tumor is detected and highlighted image is provided
//       if (response.ml_results.prediction === 'Positive' && response.highlighted_image) {
//         // Convert base64 to image URL
//         setHighlightedImageUrl(`data:image/jpeg;base64,${response.highlighted_image}`);
//       }
      
//       response.image_id = Date.now().toString(); // Generate a temporary ID for the new upload
//       response.upload_time = new Date().toISOString();
//       response.filename = selectedFile.name;
//       response.image_data = previewUrl.split(',')[1]; // Get base64 data
//       setResults({ ...response.ml_results, fullData: response });
//     } catch (err) {
//       setError(err.message || 'Failed to upload image. Please try again.');
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
//       <div className="max-w-3xl mx-auto">
//         <div className="bg-white dark:bg-gray-800 shadow-xl rounded-lg overflow-hidden">
//           <div className="px-6 py-8 border-b border-gray-200 dark:border-gray-700">
//             <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Upload Brain Scan</h1>
//             <p className="mt-2 text-gray-600 dark:text-gray-400">
//               Upload your brain scan image for AI analysis
//             </p>
//           </div>
//           <div className="px-6 py-6">
//             {error && (
//               <div className="mb-4 bg-red-50 dark:bg-red-900/50 border-l-4 border-red-400 dark:border-red-500 p-4">
//                 <div className="flex">
//                   <div className="flex-shrink-0">
//                     <svg className="h-5 w-5 text-red-400 dark:text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
//                       <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
//                     </svg>
//                   </div>
//                   <div className="ml-3">
//                     <p className="text-sm text-red-700 dark:text-red-200">{error}</p>
//                   </div>
//                 </div>
//               </div>
//             )}
//             <form onSubmit={handleUpload}>
//               <div className="space-y-6">
//                 <div>
//                   <label className="block text-sm font-medium text-gray-700 dark:text-gray-200">
//                     Brain Scan Image
//                   </label>
//                   <div className="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 dark:border-gray-600 border-dashed rounded-md">
//                     <div className="space-y-1 text-center">
//                       {previewUrl ? (
//                         <div className="flex flex-col items-center">
//                           <img
//                             src={previewUrl}
//                             alt="Preview"
//                             className="max-h-64 mb-4 rounded"
//                           />
//                           <button
//                             type="button"
//                             onClick={() => {
//                               setSelectedFile(null);
//                               setPreviewUrl(null);
//                               setResults(null);
//                               setHighlightedImageUrl(null);
//                             }}
//                             className="text-sm text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
//                           >
//                             Remove image
//                           </button>
//                         </div>
//                       ) : (
//                         <>
//                           <svg className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
//                             <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
//                           </svg>
//                           <div className="flex text-sm text-gray-600 dark:text-gray-400">
//                             <label
//                               htmlFor="file-upload"
//                               className="relative cursor-pointer bg-white dark:bg-gray-800 rounded-md font-medium text-primary-600 dark:text-primary-400 hover:text-primary-500 dark:hover:text-primary-300 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400"
//                             >
//                               <span>Upload a file</span>
//                               <input id="file-upload" name="file-upload" type="file" className="sr-only" accept="image/*" onChange={handleFileChange} />
//                             </label>
//                             <p className="pl-1">or drag and drop</p>
//                           </div>
//                           <p className="text-xs text-gray-500 dark:text-gray-400">
//                             PNG, JPG, JPEG up to 16MB
//                           </p>
//                         </>
//                       )}
//                     </div>
//                   </div>
//                 </div>

//                 <div className="flex justify-center">
//                   <button
//                     type="submit"
//                     disabled={loading || !selectedFile}
//                     className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 dark:bg-primary-500 dark:hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400 disabled:opacity-50"
//                   >
//                     {loading ? (
//                       <>
//                         <div className="w-5 h-5 mr-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
//                         Processing...
//                       </>
//                     ) : (
//                       'Analyze Image'
//                     )}
//                   </button>
//                 </div>
//               </div>
//             </form>
//           </div>

//           {/* Results Section */}
//           {results && (
//             <div className="px-6 py-6 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
//               <div className="flex justify-between items-center mb-4">
//                 <h2 className="text-xl font-bold text-gray-900 dark:text-white">Analysis Results</h2>
//               </div>
//               <div className="space-y-4">
//                 <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow">
//                   <div className="flex items-center mb-4">
//                     <div className={`w-3 h-3 rounded-full mr-2 ${results.prediction === 'Positive' ? 'bg-red-500' : 'bg-green-500'}`}></div>
//                     <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
//                       {results.prediction === 'Positive' ? 'Tumor Detected' : 'No Tumor Detected'}
//                     </h3>
//                   </div>

//                   {results.prediction === 'Positive' && (
//                     <>
//                       {/* Image Comparison Section */}
//                       {highlightedImageUrl && (
//                         <div className="mb-6">
//                           <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">Tumor Visualization</p>
//                           <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                             <div>
//                               <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 text-center">Original Image</p>
//                               <img 
//                                 src={previewUrl} 
//                                 alt="Original Brain Scan" 
//                                 className="w-full h-auto rounded border border-gray-200 dark:border-gray-700"
//                               />
//                             </div>
//                             <div>
//                               <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 text-center">Highlighted Tumor</p>
//                               <img 
//                                 src={highlightedImageUrl} 
      
                                
//                                 alt="Tumor Highlighted" 
//                                 className="w-full h-auto rounded border border-gray-200 dark:border-gray-700"
//                               />
//                             </div>
//                           </div>
//                           <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
//                             The highlighted areas indicate potential tumor regions identified by our explainable AI.
//                           </p>
//                         </div>
//                       )}
                      
//                       <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
//                         <div>
//                           <p className="text-sm text-gray-500 dark:text-gray-400">Tumor Type</p>
//                           <p className="font-medium text-gray-900 dark:text-gray-100">{results.tumor_type}</p>
//                         </div>
//                         <div>
//                           <p className="text-sm text-gray-500 dark:text-gray-400">Confidence</p>
//                           <p className="font-medium text-gray-900 dark:text-gray-100">{(results.confidence * 100).toFixed(2)}%</p>
//                         </div>
//                       </div>

//                       <div className="mb-4">
//                         <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Recommended Precautions</p>
//                         <ul className="list-disc pl-5 space-y-1">
//                           {results.precautions.map((precaution, index) => (
//                             <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{precaution}</li>
//                           ))}
//                         </ul>
//                       </div>

//                       <div>
//                         <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Treatment Options</p>
//                         <ul className="list-disc pl-5 space-y-1">
//                           {results.treatment_options.map((option, index) => (
//                             <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{option}</li>
//                           ))}
//                         </ul>
//                       </div>
//                     </>
//                   )}

//                   <div className="mt-6">
//                     <p className="text-sm text-gray-500 dark:text-gray-400">
//                       <strong className="text-gray-700 dark:text-gray-300">Important:</strong> This is an AI-assisted analysis and should not replace professional medical diagnosis. 
//                       Please consult with a healthcare professional regarding these results.
//                     </p>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Upload;

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadImage, getCurrentUser } from '../services/api';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [highlightedImageUrl, setHighlightedImageUrl] = useState(null);
  const navigate = useNavigate();
  const currentUser = getCurrentUser();

  // Redirect if not logged in
  if (!currentUser) {
    navigate('/login');
    return null;
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError('');
      setResults(null);
      setHighlightedImageUrl(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await uploadImage(selectedFile, currentUser.id);
      
      // Check if we have a highlighted image and set it
      if (response.ml_results && response.ml_results.highlighted_image) {
        setHighlightedImageUrl(`data:image/png;base64,${response.ml_results.highlighted_image}`);
      }
      
      response.image_id = Date.now().toString(); // Generate a temporary ID for the new upload
      response.upload_time = new Date().toISOString();
      response.filename = selectedFile.name;
      response.image_data = previewUrl.split(',')[1]; // Get base64 data
      setResults({ ...response.ml_results, fullData: response });
    } catch (err) {
      setError(err.message || 'Failed to upload image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white dark:bg-gray-800 shadow-xl rounded-lg overflow-hidden">
          <div className="px-6 py-8 border-b border-gray-200 dark:border-gray-700">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Upload Brain Scan</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">
              Upload your brain scan image for AI analysis
            </p>
          </div>
          <div className="px-6 py-6">
            {error && (
              <div className="mb-4 bg-red-50 dark:bg-red-900/50 border-l-4 border-red-400 dark:border-red-500 p-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400 dark:text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-red-700 dark:text-red-200">{error}</p>
                  </div>
                </div>
              </div>
            )}
            <form onSubmit={handleUpload}>
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                    Brain Scan Image
                  </label>
                  <div className="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 dark:border-gray-600 border-dashed rounded-md">
                    <div className="space-y-1 text-center">
                      {previewUrl ? (
                        <div className="flex flex-col items-center">
                          <img
                            src={previewUrl}
                            alt="Preview"
                            className="max-h-64 mb-4 rounded"
                          />
                          <button
                            type="button"
                            onClick={() => {
                              setSelectedFile(null);
                              setPreviewUrl(null);
                              setResults(null);
                              setHighlightedImageUrl(null);
                            }}
                            className="text-sm text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
                          >
                            Remove image
                          </button>
                        </div>
                      ) : (
                        <>
                          <svg className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                          <div className="flex text-sm text-gray-600 dark:text-gray-400">
                            <label
                              htmlFor="file-upload"
                              className="relative cursor-pointer bg-white dark:bg-gray-800 rounded-md font-medium text-primary-600 dark:text-primary-400 hover:text-primary-500 dark:hover:text-primary-300 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400"
                            >
                              <span>Upload a file</span>
                              <input id="file-upload" name="file-upload" type="file" className="sr-only" accept="image/*" onChange={handleFileChange} />
                            </label>
                            <p className="pl-1">or drag and drop</p>
                          </div>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            PNG, JPG, JPEG up to 16MB
                          </p>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex justify-center">
                  <button
                    type="submit"
                    disabled={loading || !selectedFile}
                    className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 dark:bg-primary-500 dark:hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400 disabled:opacity-50"
                  >
                    {loading ? (
                      <>
                        <div className="w-5 h-5 mr-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        Processing...
                      </>
                    ) : (
                      'Analyze Image'
                    )}
                  </button>
                </div>
              </div>
            </form>
          </div>

          {/* Results Section with Highlighted Image */}
          {results && (
            <div className="px-6 py-6 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Analysis Results</h2>
              </div>
              
              {/* Image Comparison Section - Only show for positive results */}
              {results.prediction === 'Positive' && highlightedImageUrl && (
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                    Tumor Visualization
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-white dark:bg-gray-900 p-4 rounded-lg shadow">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Original Scan</p>
                      <img 
                        src={previewUrl} 
                        alt="Original brain scan" 
                        className="rounded-md border border-gray-300 dark:border-gray-700"
                      />
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                        Highlighted Tumor Region
                        <span className="text-xs ml-1 text-amber-600 dark:text-amber-400">
                          (Using Explainable AI)
                        </span>
                      </p>
                      <img 
                        src={highlightedImageUrl} 
                        alt="Highlighted tumor regions" 
                        className="rounded-md border border-gray-300 dark:border-gray-700"
                      />
                    </div>
                    <div className="md:col-span-2 mt-2">
                      <p className="text-xs text-gray-500 dark:text-gray-400 italic">
                        Heatmap visualization shows areas of concern identified by our AI model. 
                        Warmer colors (red, yellow) indicate higher probability of tumor presence.
                      </p>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Regular Results Display */}
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow">
                  <div className="flex items-center mb-4">
                    <div className={`w-3 h-3 rounded-full mr-2 ${results.prediction === 'Positive' ? 'bg-red-500' : 'bg-green-500'}`}></div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      {results.prediction === 'Positive' ? 'Tumor Detected' : 'No Tumor Detected'}
                    </h3>
                  </div>

                  {results.prediction === 'Positive' && (
                    <>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div>
                          <p className="text-sm text-gray-500 dark:text-gray-400">Tumor Type</p>
                          <p className="font-medium text-gray-900 dark:text-gray-100">{results.tumor_type}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500 dark:text-gray-400">Confidence</p>
                          <p className="font-medium text-gray-900 dark:text-gray-100">{(results.confidence * 100).toFixed(2)}%</p>
                        </div>
                      </div>

                      <div className="mb-4">
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Recommended Precautions</p>
                        <ul className="list-disc pl-5 space-y-1">
                          {results.precautions.map((precaution, index) => (
                            <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{precaution}</li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Treatment Options</p>
                        <ul className="list-disc pl-5 space-y-1">
                          {results.treatment_options.map((option, index) => (
                            <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{option}</li>
                          ))}
                        </ul>
                      </div>
                    </>
                  )}

                  <div className="mt-6">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      <strong className="text-gray-700 dark:text-gray-300">Important:</strong> This is an AI-assisted analysis and should not replace professional medical diagnosis. 
                      Please consult with a healthcare professional regarding these results.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Upload;