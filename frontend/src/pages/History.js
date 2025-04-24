import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getHistory, getCurrentUser, addStarredImage } from '../services/api';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [tickedImages, setTickedImages] = useState({});
  const [noteText, setNoteText] = useState('');
  const [showNoteModal, setShowNoteModal] = useState(false);
  const [currentTickedImage, setCurrentTickedImage] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const navigate = useNavigate();
  const currentUser = getCurrentUser();

  useEffect(() => {
    if (!currentUser) {
      navigate('/login');
      return;
    }

    const fetchHistory = async () => {
      try {
        const historyData = await getHistory(currentUser.id);
        setHistory(historyData);
        // Load starred images from localStorage
        const savedImages = JSON.parse(localStorage.getItem('starredImages') || '{}');
        setTickedImages(savedImages);
      } catch (err) {
        setError('Failed to load history. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [currentUser, navigate]);

  const handleImageClick = (image) => {
    setSelectedImage(image);
  };

  const handleTickImage = (image) => {
    if (tickedImages[image.image_id]) {
      // If already ticked, show edit modal
      setCurrentTickedImage(image);
      setNoteText(tickedImages[image.image_id].note || '');
      setIsEditing(true);
      setShowNoteModal(true);
    } else {
      // If not ticked, show new note modal
      setCurrentTickedImage(image);
      setNoteText('');
      setIsEditing(false);
      setShowNoteModal(true);
    }
  };

  const handleAddNote = async () => {
    if (isSaving) return;
    setIsSaving(true);

    try {
      if (noteText.trim()) {
        const currentUser = getCurrentUser();
        if (!currentUser) {
          navigate('/login');
          return;
        }

        // Make sure we have the complete image data
        const imageToSave = {
          ...currentTickedImage,
          note: noteText,
          timestamp: isEditing ? tickedImages[currentTickedImage.image_id].timestamp : new Date().toISOString(),
          image_data: currentTickedImage.image_data || (selectedImage && selectedImage.image_id === currentTickedImage.image_id ? selectedImage.image_data : null)
        };
        
        // Log for debugging
        console.log('Saving image data:', {
          hasImageData: !!imageToSave.image_data,
          imageId: imageToSave.image_id
        });
        
        // Save to MongoDB
        await addStarredImage(currentUser.id, imageToSave.image_id, noteText);
        
        // Update local state for UI
        const updatedImages = {
          ...tickedImages,
          [currentTickedImage.image_id]: imageToSave
        };
        
        setTickedImages(updatedImages);
        
        // Also keep localStorage for compatibility with existing code
        localStorage.setItem('starredImages', JSON.stringify(updatedImages));
      }
    } catch (error) {
      console.error('Error saving starred image:', error);
      alert('Failed to save starred image. Please try again.');
    } finally {
      setShowNoteModal(false);
      setNoteText('');
      setCurrentTickedImage(null);
      setIsEditing(false);
      setIsSaving(false);
    }
  };

  const handleCloseModal = () => {
    setShowNoteModal(false);
    setNoteText('');
    setCurrentTickedImage(null);
  };

  const formatDate = (dateString) => {
    const options = { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  // Note Modal Component
  const NoteModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={handleCloseModal}>
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-96 max-w-[90vw]" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {!isEditing ? 'Add Star Note' : 'Edit Star Note'}
          </h3>
          <button
            onClick={handleCloseModal}
            className="text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
          >
            <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <textarea
          id="note-text"
          className="w-full h-32 p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:focus:ring-primary-400 dark:focus:border-primary-400 resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-white border-gray-300 dark:border-gray-600 placeholder-gray-400 dark:placeholder-gray-500"
          placeholder="Enter your note here..."
          value={noteText}
          onChange={(e) => {
            const newValue = e.target.value;
            requestAnimationFrame(() => {
              e.target.selectionStart = e.target.value.length;
              e.target.selectionEnd = e.target.value.length;
            });
            setNoteText(newValue);
          }}
          autoFocus
          dir="ltr"
          spellCheck="false"
          style={{ 
            textAlign: 'left',
            caretColor: 'auto'
          }}
          onFocus={(e) => {
            e.target.selectionStart = e.target.value.length;
            e.target.selectionEnd = e.target.value.length;
          }}
        />
        <div className="flex justify-end space-x-2">
          <button
            type="button"
            onClick={handleCloseModal}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
            disabled={isSaving}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleAddNote}
            className="px-4 py-2 text-sm font-medium text-white bg-primary-600 dark:bg-primary-500 rounded-md hover:bg-primary-700 dark:hover:bg-primary-600"
            disabled={!noteText.trim() || isSaving}
          >
            {isSaving ? 'Saving...' : !isEditing ? 'Save Note' : 'Update Note'}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      {showNoteModal && <NoteModal />}
      <div className="max-w-6xl mx-auto">
        <div className="bg-white dark:bg-gray-800 shadow-xl rounded-lg overflow-hidden">
          <div className="px-6 py-8 border-b border-gray-200 dark:border-gray-700">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Scan History</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">
              View your previous brain scan analyses and results
            </p>
          </div>

          {loading ? (
            <div className="flex justify-center items-center p-12">
              <div className="loader w-12 h-12 border-4 border-gray-200 dark:border-gray-700 rounded-full"></div>
            </div>
          ) : error ? (
            <div className="p-6 text-center">
              <p className="text-red-500 dark:text-red-400">{error}</p>
            </div>
          ) : history.length === 0 ? (
            <div className="p-6 text-center">
              <p className="text-gray-500 dark:text-gray-400">No scan history found. Upload your first scan to get started.</p>
            </div>
          ) : (
            <div className="flex flex-col md:flex-row">
              {/* Details view */}
              <div className="w-full md:w-2/3">
                {selectedImage ? (
                  <div>
                    <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between items-center">
                        <h2 className="text-xl font-bold text-gray-900 dark:text-white">{selectedImage.filename}</h2>
                        <button
                          onClick={() => handleTickImage(selectedImage)}
                          className="p-2 text-gray-400 hover:text-yellow-500 dark:text-gray-500 dark:hover:text-yellow-400 transition-colors duration-200"
                          title={tickedImages[selectedImage.image_id] ? "Edit star note" : "Add star note"}
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill={tickedImages[selectedImage.image_id] ? "currentColor" : "none"} viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                          </svg>
                        </button>
                      </div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Uploaded on {formatDate(selectedImage.upload_time)}
                      </p>
                      <div className="p-6">
                        

                        {selectedImage.ml_results.tumor_detected && (
                          <>
                            <div className="mb-4">
                              <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Recommended Precautions</p>
                              <ul className="list-disc pl-5 space-y-1">
                                {selectedImage.ml_results.precautions.map((precaution, index) => (
                                  <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{precaution}</li>
                                ))}
                              </ul>
                            </div>

                            <div>
                              <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Treatment Options</p>
                              <ul className="list-disc pl-5 space-y-1">
                                {selectedImage.ml_results.treatment_options.map((option, index) => (
                                  <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{option}</li>
                                ))}
                              </ul>
                            </div>
                          </>
                        )}

                        {/* <div className="mt-6">
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            <strong className="text-gray-700 dark:text-gray-300">Important:</strong> This is an AI-assisted analysis and should not replace 
                            professional medical diagnosis. Please consult with a healthcare professional 
                            regarding these results.
                          </p>
                        </div> */}
                      </div>
                    </div>
                    <div className="p-6">
                      <div className="flex flex-col md:flex-row gap-6">
                        <div className="w-full md:w-1/2">
                          <div className="bg-black rounded-lg overflow-hidden shadow-lg">
                            <img
                              src={`data:image/jpeg;base64,${selectedImage.image_data}`}
                              alt="Brain scan"
                              className="w-full object-contain"
                              style={{ maxHeight: '400px' }}
                            />
                          </div>
                        </div>
                        <div className="w-full md:w-1/2">
                          <div className="bg-white dark:bg-gray-900 p-4 rounded-lg shadow-md">
                            <div className="flex items-center mb-4">
                              <div 
                                className={`w-3 h-3 rounded-full mr-2 ${
                                  selectedImage.ml_results.prediction === 'Positive' ? 'bg-red-500' : 'bg-green-500'
                                }`}
                              ></div>
                              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                                {selectedImage.ml_results.prediction === 'Positive' ? 'Tumor Detected' : 'No Tumor Detected'}
                              </h3>
                            </div>

                            {/* {selectedImage.ml_results.prediction === 'Positive' && (
                              <>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                                  <div>
                                    <p className="text-sm text-gray-500 dark:text-gray-400">Tumor Type</p>
                                    <p className="font-medium text-gray-900 dark:text-gray-100">{selectedImage.ml_results.tumor_type}</p>
                                  </div>
                                  <div>
                                    <p className="text-sm text-gray-500 dark:text-gray-400">Confidence</p>
                                    <p className="font-medium text-gray-900 dark:text-gray-100">{(selectedImage.ml_results.confidence * 100).toFixed(2)}%</p>
                                  </div>
                                </div>

                                <div className="mb-4">
                                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Recommended Precautions</p>
                                  <ul className="list-disc pl-5 space-y-1">
                                    {selectedImage.ml_results.precautions.map((precaution, index) => (
                                      <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{precaution}</li>
                                    ))}
                                  </ul>
                                </div>

                                <div>
                                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Treatment Options</p>
                                  <ul className="list-disc pl-5 space-y-1">
                                    {selectedImage.ml_results.treatment_options.map((option, index) => (
                                      <li key={index} className="text-sm text-gray-700 dark:text-gray-300">{option}</li>
                                    ))}
                                  </ul>
                                </div>
                              </>
                            )} */}

                            {/* <div className="mt-6">
                              <p className="text-sm text-gray-500 dark:text-gray-400">
                                <strong className="text-gray-700 dark:text-gray-300">Important:</strong> This is an AI-assisted analysis and should not replace 
                                professional medical diagnosis. Please consult with a healthcare professional 
                                regarding these results.
                              </p>
                            </div> */}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full min-h-[400px]">
                    <div className="text-center p-6">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      <p className="text-gray-500">Select a scan from the list to view details</p>
                    </div>
                  </div>
                )}
              </div>

              {/* List of scans */}
              <div className="w-full md:w-1/3 border-l border-gray-200 dark:border-gray-700">
                <div className="p-4 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                  <h2 className="text-3xl font-bold text-primary-500 dark:text-primary-400">Your Scans</h2>
                </div>
                <div className="overflow-y-auto dark:bg-gray-800" style={{ maxHeight: '70vh' }}>
                  {history.map((item) => (
                    <div
                      key={item.image_id}
                      className={`relative p-4 border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors duration-150 ${
                        selectedImage && selectedImage.image_id === item.image_id
                          ? 'bg-primary-50 dark:bg-primary-900/20 border-l-4 border-primary-500 dark:border-primary-400'
                          : ''
                      }`}
                      onClick={() => handleImageClick(item)}
                    >
                      {tickedImages[item.image_id] && (
                        <div className="absolute top-2 right-2 bg-white dark:bg-gray-800 rounded-full p-1 shadow-lg">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        </div>
                      )}
                      <div className="flex items-start">
                        <div className="h-10 w-10 flex-shrink-0 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden relative">
                          {tickedImages[item.image_id] && (
                            <div className="absolute top-0 right-0 bg-white dark:bg-gray-800 rounded-full p-0.5 shadow-md z-10">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                              </svg>
                            </div>
                          )}
                          <img
                            src={`data:image/jpeg;base64,${item.image_data}`}
                            alt="Scan thumbnail"
                            className="h-full w-full object-cover"
                          />
                        </div>
                        <div className="ml-3">
                          <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                            {item.filename}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {formatDate(item.upload_time)}
                          </p>
                          {tickedImages[item.image_id] && (
                            <div className="mt-1 bg-gray-50 dark:bg-gray-700 rounded-md p-2">
                              <p className="text-xs text-gray-700 dark:text-gray-300">
                                <span className="truncate">{tickedImages[item.image_id].note}</span>
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default History; 